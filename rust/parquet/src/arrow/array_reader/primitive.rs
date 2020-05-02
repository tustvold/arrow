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

use std::marker::PhantomData;

use std::result::Result::Ok;

use arrow::array::ArrayRef;
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType as ArrowType, IntervalUnit};

use crate::arrow::converter::{
    BoolConverter, Converter, Float32Converter, Float64Converter, Int16Converter,
    Int32Converter, Int64Converter, Int8Converter, UInt16Converter, UInt32Converter,
    UInt64Converter, UInt8Converter,
};
use crate::arrow::record_reader::RecordReader;
use crate::arrow::schema::parquet_to_arrow_field;
use crate::basic::Type as PhysicalType;
use crate::column::page::PageIterator;

use crate::data_type::{BoolType, DataType, DoubleType, FloatType, Int32Type, Int64Type};
use crate::errors::{ParquetError, Result};

use crate::schema::types::ColumnDescPtr;

use crate::arrow::array_reader::ArrayReader;
use std::any::Any;

/// Primitive array readers are leaves of array reader tree. They accept page iterator
/// and read them into primitive arrays.
pub struct PrimitiveArrayReader<T: DataType> {
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Buffer>,
    rep_levels_buffer: Option<Buffer>,
    column_desc: ColumnDescPtr,
    record_reader: RecordReader<T>,
    _type_marker: PhantomData<T>,
}

impl<T: DataType> PrimitiveArrayReader<T> {
    /// Construct primitive array reader.
    pub fn new(
        mut pages: Box<dyn PageIterator>,
        column_desc: ColumnDescPtr,
    ) -> Result<Self> {
        let data_type = parquet_to_arrow_field(column_desc.as_ref())?
            .data_type()
            .clone();

        let mut record_reader = RecordReader::<T>::new(column_desc.clone());
        record_reader.set_page_reader(
            pages
                .next()
                .ok_or_else(|| general_err!("Can't build array without pages!"))??,
        )?;

        Ok(Self {
            data_type,
            pages,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            column_desc,
            record_reader,
            _type_marker: PhantomData,
        })
    }
}

/// Implementation of primitive array reader.
impl<T: DataType> ArrayReader for PrimitiveArrayReader<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns data type of primitive array.
    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    /// Reads at most `batch_size` records into array.
    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef> {
        let mut records_read = 0usize;
        while records_read < batch_size {
            let records_to_read = batch_size - records_read;

            let records_read_once = self.record_reader.read_records(records_to_read)?;
            records_read = records_read + records_read_once;

            // Record reader exhausted
            if records_read_once < records_to_read {
                if let Some(page_reader) = self.pages.next() {
                    // Read from new page reader
                    self.record_reader.set_page_reader(page_reader?)?;
                } else {
                    // Page reader also exhausted
                    break;
                }
            }
        }

        // convert to arrays
        let array = match (&self.data_type, T::get_physical_type()) {
            (ArrowType::Boolean, PhysicalType::BOOLEAN) => {
                BoolConverter::convert(self.record_reader.cast::<BoolType>())
            }
            (ArrowType::Int8, PhysicalType::INT32) => {
                Int8Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Int16, PhysicalType::INT32) => {
                Int16Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Int32, PhysicalType::INT32) => {
                Int32Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::UInt8, PhysicalType::INT32) => {
                UInt8Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::UInt16, PhysicalType::INT32) => {
                UInt16Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::UInt32, PhysicalType::INT32) => {
                UInt32Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Int64, PhysicalType::INT64) => {
                Int64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::UInt64, PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::Float32, PhysicalType::FLOAT) => {
                Float32Converter::convert(self.record_reader.cast::<FloatType>())
            }
            (ArrowType::Float64, PhysicalType::DOUBLE) => {
                Float64Converter::convert(self.record_reader.cast::<DoubleType>())
            }
            (ArrowType::Timestamp(_, _), PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::Date32(_), PhysicalType::INT32) => {
                UInt32Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Date64(_), PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::Time32(_), PhysicalType::INT32) => {
                UInt32Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Time64(_), PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::Interval(IntervalUnit::YearMonth), PhysicalType::INT32) => {
                UInt32Converter::convert(self.record_reader.cast::<Int32Type>())
            }
            (ArrowType::Interval(IntervalUnit::DayTime), PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (ArrowType::Duration(_), PhysicalType::INT64) => {
                UInt64Converter::convert(self.record_reader.cast::<Int64Type>())
            }
            (arrow_type, physical_type) => Err(general_err!(
                "Reading {:?} type from parquet {:?} is not supported yet.",
                arrow_type,
                physical_type
            )),
        }?;

        // save definition and repetition buffers
        self.def_levels_buffer = self.record_reader.consume_def_levels()?;
        self.rep_levels_buffer = self.record_reader.consume_rep_levels()?;
        self.record_reader.reset();
        Ok(array)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer
            .as_ref()
            .map(|buf| unsafe { buf.typed_data() })
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer
            .as_ref()
            .map(|buf| unsafe { buf.typed_data() })
    }
}
