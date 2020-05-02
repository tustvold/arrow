// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::{HashMap, HashSet};

use std::rc::Rc;
use std::result::Result::Ok;

use std::vec::Vec;

use arrow::datatypes::{DataType as ArrowType, Field};

use crate::arrow::converter::{BinaryConverter, Int96Converter, Utf8Converter};

use crate::basic::{LogicalType, Repetition, Type as PhysicalType};

use crate::data_type::{
    BoolType, ByteArrayType, DoubleType, FloatType, Int32Type, Int64Type, Int96Type,
};
use crate::errors::{ParquetError, ParquetError::ArrowError, Result};
use crate::file::reader::{FilePageIterator, FileReader};
use crate::schema::types::{ColumnDescriptor, ColumnPath, SchemaDescPtr, Type, TypePtr};
use crate::schema::visitor::TypeVisitor;

use crate::arrow::array_reader::{
    ArrayReader, ComplexObjectArrayReader, PrimitiveArrayReader, StructArrayReader,
};

/// Create array reader from parquet schema, column indices, and parquet file reader.
pub fn build_array_reader<T>(
    parquet_schema: SchemaDescPtr,
    column_indices: T,
    file_reader: Rc<dyn FileReader>,
) -> Result<Box<dyn ArrayReader>>
where
    T: IntoIterator<Item = usize>,
{
    let mut base_nodes = Vec::new();
    let mut base_nodes_set = HashSet::new();
    let mut leaves = HashMap::<*const Type, usize>::new();

    for c in column_indices {
        let column = parquet_schema.column(c).self_type() as *const Type;
        let root = parquet_schema.get_column_root_ptr(c);
        let root_raw_ptr = root.clone().as_ref() as *const Type;

        leaves.insert(column, c);
        if !base_nodes_set.contains(&root_raw_ptr) {
            base_nodes.push(root);
            base_nodes_set.insert(root_raw_ptr);
        }
    }

    if leaves.is_empty() {
        return Err(general_err!("Can't build array reader without columns!"));
    }

    ArrayReaderBuilder::new(
        Rc::new(parquet_schema.root_schema().clone()),
        Rc::new(leaves),
        file_reader,
    )
    .build_array_reader()
}

/// Used to build array reader.
struct ArrayReaderBuilder {
    root_schema: TypePtr,
    // Key: columns that need to be included in final array builder
    // Value: column index in schema
    columns_included: Rc<HashMap<*const Type, usize>>,
    file_reader: Rc<dyn FileReader>,
}

/// Used in type visitor.
#[derive(Clone)]
struct ArrayReaderBuilderContext {
    def_level: i16,
    rep_level: i16,
    path: ColumnPath,
}

impl Default for ArrayReaderBuilderContext {
    fn default() -> Self {
        Self {
            def_level: 0i16,
            rep_level: 0i16,
            path: ColumnPath::new(Vec::new()),
        }
    }
}

/// Create array reader by visiting schema.
impl<'a> TypeVisitor<Option<Box<dyn ArrayReader>>, &'a ArrayReaderBuilderContext>
    for ArrayReaderBuilder
{
    /// Build array reader for primitive type.
    /// Currently we don't have a list reader implementation, so repeated type is not
    /// supported yet.
    fn visit_primitive(
        &mut self,
        cur_type: TypePtr,
        context: &'a ArrayReaderBuilderContext,
    ) -> Result<Option<Box<dyn ArrayReader>>> {
        if self.is_included(cur_type.as_ref()) {
            let mut new_context = context.clone();
            new_context.path.append(vec![cur_type.name().to_string()]);

            match cur_type.get_basic_info().repetition() {
                Repetition::REPEATED => {
                    new_context.def_level += 1;
                    new_context.rep_level += 1;
                }
                Repetition::OPTIONAL => {
                    new_context.def_level += 1;
                }
                _ => (),
            }

            let reader =
                self.build_for_primitive_type_inner(cur_type.clone(), &new_context)?;

            if cur_type.get_basic_info().repetition() == Repetition::REPEATED {
                Err(ArrowError(
                    "Reading repeated field is not supported yet!".to_string(),
                ))
            } else {
                Ok(Some(reader))
            }
        } else {
            Ok(None)
        }
    }

    /// Build array reader for struct type.
    fn visit_struct(
        &mut self,
        cur_type: Rc<Type>,
        context: &'a ArrayReaderBuilderContext,
    ) -> Result<Option<Box<ArrayReader>>> {
        let mut new_context = context.clone();
        new_context.path.append(vec![cur_type.name().to_string()]);

        if cur_type.get_basic_info().has_repetition() {
            match cur_type.get_basic_info().repetition() {
                Repetition::REPEATED => {
                    new_context.def_level += 1;
                    new_context.rep_level += 1;
                }
                Repetition::OPTIONAL => {
                    new_context.def_level += 1;
                }
                _ => (),
            }
        }

        if let Some(reader) = self.build_for_struct_type_inner(&cur_type, &new_context)? {
            if cur_type.get_basic_info().has_repetition()
                && cur_type.get_basic_info().repetition() == Repetition::REPEATED
            {
                Err(ArrowError(
                    "Reading repeated field is not supported yet!".to_string(),
                ))
            } else {
                Ok(Some(reader))
            }
        } else {
            Ok(None)
        }
    }

    /// Build array reader for map type.
    /// Currently this is not supported.
    fn visit_map(
        &mut self,
        _cur_type: Rc<Type>,
        _context: &'a ArrayReaderBuilderContext,
    ) -> Result<Option<Box<dyn ArrayReader>>> {
        Err(ArrowError(
            "Reading parquet map array into arrow is not supported yet!".to_string(),
        ))
    }

    /// Build array reader for list type.
    /// Currently this is not supported.
    fn visit_list_with_item(
        &mut self,
        _list_type: Rc<Type>,
        _item_type: &Type,
        _context: &'a ArrayReaderBuilderContext,
    ) -> Result<Option<Box<dyn ArrayReader>>> {
        Err(ArrowError(
            "Reading parquet list array into arrow is not supported yet!".to_string(),
        ))
    }
}

impl<'a> ArrayReaderBuilder {
    /// Construct array reader builder.
    fn new(
        root_schema: TypePtr,
        columns_included: Rc<HashMap<*const Type, usize>>,
        file_reader: Rc<dyn FileReader>,
    ) -> Self {
        Self {
            root_schema,
            columns_included,
            file_reader,
        }
    }

    /// Main entry point.
    fn build_array_reader(&mut self) -> Result<Box<dyn ArrayReader>> {
        let context = ArrayReaderBuilderContext::default();

        self.visit_struct(self.root_schema.clone(), &context)
            .and_then(|reader_opt| {
                reader_opt.ok_or_else(|| general_err!("Failed to build array reader!"))
            })
    }

    // Utility functions

    /// Check whether one column in included in this array reader builder.
    fn is_included(&self, t: &Type) -> bool {
        self.columns_included.contains_key(&(t as *const Type))
    }

    /// Creates primitive array reader for each primitive type.
    fn build_for_primitive_type_inner(
        &self,
        cur_type: TypePtr,
        context: &'a ArrayReaderBuilderContext,
    ) -> Result<Box<dyn ArrayReader>> {
        let column_desc = Rc::new(ColumnDescriptor::new(
            cur_type.clone(),
            Some(self.root_schema.clone()),
            context.def_level,
            context.rep_level,
            context.path.clone(),
        ));
        let page_iterator = Box::new(FilePageIterator::new(
            self.columns_included[&(cur_type.as_ref() as *const Type)],
            self.file_reader.clone(),
        )?);

        match cur_type.get_physical_type() {
            PhysicalType::BOOLEAN => Ok(Box::new(PrimitiveArrayReader::<BoolType>::new(
                page_iterator,
                column_desc,
            )?)),
            PhysicalType::INT32 => Ok(Box::new(PrimitiveArrayReader::<Int32Type>::new(
                page_iterator,
                column_desc,
            )?)),
            PhysicalType::INT64 => Ok(Box::new(PrimitiveArrayReader::<Int64Type>::new(
                page_iterator,
                column_desc,
            )?)),
            PhysicalType::INT96 => {
                Ok(Box::new(ComplexObjectArrayReader::<
                    Int96Type,
                    Int96Converter,
                >::new(page_iterator, column_desc)?))
            }
            PhysicalType::FLOAT => Ok(Box::new(PrimitiveArrayReader::<FloatType>::new(
                page_iterator,
                column_desc,
            )?)),
            PhysicalType::DOUBLE => Ok(Box::new(
                PrimitiveArrayReader::<DoubleType>::new(page_iterator, column_desc)?,
            )),
            PhysicalType::BYTE_ARRAY => {
                if cur_type.get_basic_info().logical_type() == LogicalType::UTF8 {
                    Ok(Box::new(ComplexObjectArrayReader::<
                        ByteArrayType,
                        Utf8Converter,
                    >::new(
                        page_iterator, column_desc
                    )?))
                } else {
                    Ok(Box::new(ComplexObjectArrayReader::<
                        ByteArrayType,
                        BinaryConverter,
                    >::new(
                        page_iterator, column_desc
                    )?))
                }
            }
            other => Err(ArrowError(format!(
                "Unable to create primitive array reader for parquet physical type {}",
                other
            ))),
        }
    }

    /// Constructs struct array reader without considering repetition.
    fn build_for_struct_type_inner(
        &mut self,
        cur_type: &Type,
        context: &'a ArrayReaderBuilderContext,
    ) -> Result<Option<Box<dyn ArrayReader>>> {
        let mut fields = Vec::with_capacity(cur_type.get_fields().len());
        let mut children_reader = Vec::with_capacity(cur_type.get_fields().len());

        for child in cur_type.get_fields() {
            if let Some(child_reader) = self.dispatch(child.clone(), context)? {
                fields.push(Field::new(
                    child.name(),
                    child_reader.get_data_type().clone(),
                    child.is_optional(),
                ));
                children_reader.push(child_reader);
            }
        }

        if !fields.is_empty() {
            let arrow_type = ArrowType::Struct(fields);
            Ok(Some(Box::new(StructArrayReader::new(
                arrow_type,
                children_reader,
                context.def_level,
                context.rep_level,
            ))))
        } else {
            Ok(None)
        }
    }
}
