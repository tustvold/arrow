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

use arrow::array::ArrayRef;

use arrow::datatypes::DataType as ArrowType;

use crate::errors::Result;

use std::any::Any;

mod builder;
mod complex_object;
mod primitive;
mod structure;

pub use builder::build_array_reader;
pub use complex_object::ComplexObjectArrayReader;
pub use primitive::PrimitiveArrayReader;
pub use structure::StructArrayReader;

/// Array reader reads parquet data into arrow array.
pub trait ArrayReader {
    fn as_any(&self) -> &dyn Any;

    /// Returns the arrow type of this array reader.
    fn get_data_type(&self) -> &ArrowType;

    /// Reads at most `batch_size` records into an arrow array and return it.
    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef>;

    /// Returns the definition levels of data from last call of `next_batch`.
    /// The result is used by parent array reader to calculate its own definition
    /// levels and repetition levels, so that its parent can calculate null bitmap.
    fn get_def_levels(&self) -> Option<&[i16]>;

    /// Return the repetition levels of data from last call of `next_batch`.
    /// The result is used by parent array reader to calculate its own definition
    /// levels and repetition levels, so that its parent can calculate null bitmap.
    fn get_rep_levels(&self) -> Option<&[i16]>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrow::converter::Utf8Converter;
    use crate::basic::{Encoding, Type as PhysicalType};
    use crate::column::page::Page;
    use crate::data_type::{ByteArray, ByteArrayType, DataType, Int32Type, Int64Type};
    use crate::errors::Result;
    use crate::file::reader::{FileReader, SerializedFileReader};
    use crate::schema::parser::parse_message_type;
    use crate::schema::types::{ColumnDescPtr, SchemaDescriptor};
    use crate::util::test_common::page_util::{
        DataPageBuilder, DataPageBuilderImpl, InMemoryPageIterator,
    };
    use crate::util::test_common::{get_test_file, make_pages};
    use arrow::array::{Array, ArrayRef, PrimitiveArray, StringArray, StructArray};
    use arrow::datatypes::{
        DataType as ArrowType, Field, Int32Type as ArrowInt32, UInt32Type as ArrowUInt32,
        UInt64Type as ArrowUInt64,
    };
    use rand::distributions::uniform::SampleUniform;
    use rand::{thread_rng, Rng};
    use std::any::Any;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::sync::Arc;

    fn make_column_chuncks<T: DataType>(
        column_desc: ColumnDescPtr,
        encoding: Encoding,
        num_levels: usize,
        min_value: T::T,
        max_value: T::T,
        def_levels: &mut Vec<i16>,
        rep_levels: &mut Vec<i16>,
        values: &mut Vec<T::T>,
        page_lists: &mut Vec<Vec<Page>>,
        use_v2: bool,
        num_chuncks: usize,
    ) where
        T::T: PartialOrd + SampleUniform + Copy,
    {
        for _i in 0..num_chuncks {
            let mut pages = VecDeque::new();
            let mut data = Vec::new();
            let mut page_def_levels = Vec::new();
            let mut page_rep_levels = Vec::new();

            make_pages::<T>(
                column_desc.clone(),
                encoding,
                1,
                num_levels,
                min_value,
                max_value,
                &mut page_def_levels,
                &mut page_rep_levels,
                &mut data,
                &mut pages,
                use_v2,
            );

            def_levels.append(&mut page_def_levels);
            rep_levels.append(&mut page_rep_levels);
            values.append(&mut data);
            page_lists.push(Vec::from(pages));
        }
    }

    #[test]
    fn test_primitive_array_reader_data() {
        // Construct column schema
        let message_type = "
        message test_schema {
          REQUIRED INT32 leaf;
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Rc::new(SchemaDescriptor::new(Rc::new(t))))
            .unwrap();

        let column_desc = schema.column(0);

        // Construct page iterator
        {
            let mut data = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chuncks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                1,
                200,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut data,
                &mut page_lists,
                true,
                2,
            );
            let page_iterator = InMemoryPageIterator::new(
                schema.clone(),
                column_desc.clone(),
                page_lists,
            );

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc.clone(),
            )
            .unwrap();

            // Read first 50 values, which are all from the first column chunck
            let array = array_reader.next_batch(50).unwrap();
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<ArrowInt32>>()
                .unwrap();

            assert_eq!(
                &PrimitiveArray::<ArrowInt32>::from(
                    data[0..50].iter().cloned().collect::<Vec<i32>>()
                ),
                array
            );

            // Read next 100 values, the first 50 ones are from the first column chunk,
            // and the last 50 ones are from the second column chunk
            let array = array_reader.next_batch(100).unwrap();
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<ArrowInt32>>()
                .unwrap();

            assert_eq!(
                &PrimitiveArray::<ArrowInt32>::from(
                    data[50..150].iter().cloned().collect::<Vec<i32>>()
                ),
                array
            );

            // Try to read 100 values, however there are only 50 values
            let array = array_reader.next_batch(100).unwrap();
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<ArrowInt32>>()
                .unwrap();

            assert_eq!(
                &PrimitiveArray::<ArrowInt32>::from(
                    data[150..200].iter().cloned().collect::<Vec<i32>>()
                ),
                array
            );
        }
    }

    macro_rules! test_primitive_array_reader_one_type {
        ($arrow_parquet_type:ty, $physical_type:expr, $logical_type_str:expr, $result_arrow_type:ty, $result_primitive_type:ty) => {{
            let message_type = format!(
                "
            message test_schema {{
              REQUIRED {:?} leaf ({});
          }}
            ",
                $physical_type, $logical_type_str
            );
            let schema = parse_message_type(&message_type)
                .map(|t| Rc::new(SchemaDescriptor::new(Rc::new(t))))
                .unwrap();

            let column_desc = schema.column(0);

            // Construct page iterator
            {
                let mut data = Vec::new();
                let mut page_lists = Vec::new();
                make_column_chuncks::<$arrow_parquet_type>(
                    column_desc.clone(),
                    Encoding::PLAIN,
                    100,
                    1,
                    200,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    &mut data,
                    &mut page_lists,
                    true,
                    2,
                );
                let page_iterator = InMemoryPageIterator::new(
                    schema.clone(),
                    column_desc.clone(),
                    page_lists,
                );
                let mut array_reader = PrimitiveArrayReader::<$arrow_parquet_type>::new(
                    Box::new(page_iterator),
                    column_desc.clone(),
                )
                .unwrap();

                let array = array_reader.next_batch(50).unwrap();

                let array = array
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$result_arrow_type>>()
                    .unwrap();

                assert_eq!(
                    &PrimitiveArray::<$result_arrow_type>::from(
                        data[0..50]
                            .iter()
                            .map(|x| *x as $result_primitive_type)
                            .collect::<Vec<$result_primitive_type>>()
                    ),
                    array
                );
            }
        }};
    }

    #[test]
    fn test_primitive_array_reader_temporal_types() {
        test_primitive_array_reader_one_type!(
            Int32Type,
            PhysicalType::INT32,
            "DATE",
            ArrowUInt32,
            u32
        );
        test_primitive_array_reader_one_type!(
            Int32Type,
            PhysicalType::INT32,
            "TIME_MILLIS",
            ArrowUInt32,
            u32
        );
        test_primitive_array_reader_one_type!(
            Int64Type,
            PhysicalType::INT64,
            "TIME_MICROS",
            ArrowUInt64,
            u64
        );
        test_primitive_array_reader_one_type!(
            Int64Type,
            PhysicalType::INT64,
            "TIMESTAMP_MILLIS",
            ArrowUInt64,
            u64
        );
        test_primitive_array_reader_one_type!(
            Int64Type,
            PhysicalType::INT64,
            "TIMESTAMP_MICROS",
            ArrowUInt64,
            u64
        );
    }

    #[test]
    fn test_primitive_array_reader_def_and_rep_levels() {
        // Construct column schema
        let message_type = "
        message test_schema {
            REPEATED Group test_mid {
                OPTIONAL INT32 leaf;
            }
        }
        ";

        let schema = parse_message_type(message_type)
            .map(|t| Rc::new(SchemaDescriptor::new(Rc::new(t))))
            .unwrap();

        let column_desc = schema.column(0);

        // Construct page iterator
        {
            let mut def_levels = Vec::new();
            let mut rep_levels = Vec::new();
            let mut page_lists = Vec::new();
            make_column_chuncks::<Int32Type>(
                column_desc.clone(),
                Encoding::PLAIN,
                100,
                1,
                200,
                &mut def_levels,
                &mut rep_levels,
                &mut Vec::new(),
                &mut page_lists,
                true,
                2,
            );

            let page_iterator = InMemoryPageIterator::new(
                schema.clone(),
                column_desc.clone(),
                page_lists,
            );

            let mut array_reader = PrimitiveArrayReader::<Int32Type>::new(
                Box::new(page_iterator),
                column_desc.clone(),
            )
            .unwrap();

            let mut accu_len: usize = 0;

            // Read first 50 values, which are all from the first column chunck
            let array = array_reader.next_batch(50).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
            accu_len += array.len();

            // Read next 100 values, the first 50 ones are from the first column chunk,
            // and the last 50 ones are from the second column chunk
            let array = array_reader.next_batch(100).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
            accu_len += array.len();

            // Try to read 100 values, however there are only 50 values
            let array = array_reader.next_batch(100).unwrap();
            assert_eq!(
                Some(&def_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_def_levels()
            );
            assert_eq!(
                Some(&rep_levels[accu_len..(accu_len + array.len())]),
                array_reader.get_rep_levels()
            );
        }
    }

    #[test]
    fn test_complex_array_reader_def_and_rep_levels() {
        // Construct column schema
        let message_type = "
        message test_schema {
            REPEATED Group test_mid {
                OPTIONAL BYTE_ARRAY leaf (UTF8);
            }
        }
        ";
        let num_pages = 2;
        let values_per_page = 100;
        let str_base = "Hello World";

        let schema = parse_message_type(message_type)
            .map(|t| Rc::new(SchemaDescriptor::new(Rc::new(t))))
            .unwrap();

        let max_def_level = schema.column(0).max_def_level();
        let max_rep_level = schema.column(0).max_rep_level();

        assert_eq!(max_def_level, 2);
        assert_eq!(max_rep_level, 1);

        let mut rng = thread_rng();
        let column_desc = schema.column(0);
        let mut pages: Vec<Vec<Page>> = Vec::new();

        let mut rep_levels = Vec::with_capacity(num_pages * values_per_page);
        let mut def_levels = Vec::with_capacity(num_pages * values_per_page);
        let mut all_values = Vec::with_capacity(num_pages * values_per_page);

        for i in 0..num_pages {
            let mut values = Vec::with_capacity(values_per_page);

            for _ in 0..values_per_page {
                let def_level = rng.gen_range(0, max_def_level + 1);
                let rep_level = rng.gen_range(0, max_rep_level + 1);
                if def_level == max_def_level {
                    let len = rng.gen_range(1, str_base.len());
                    let slice = &str_base[..len];
                    values.push(ByteArray::from(slice));
                    all_values.push(Some(slice.to_string()));
                } else {
                    all_values.push(None)
                }
                rep_levels.push(rep_level);
                def_levels.push(def_level)
            }

            let range = i * values_per_page..(i + 1) * values_per_page;
            let mut pb =
                DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);

            pb.add_rep_levels(max_rep_level, &rep_levels.as_slice()[range.clone()]);
            pb.add_def_levels(max_def_level, &def_levels.as_slice()[range]);
            pb.add_values::<ByteArrayType>(Encoding::PLAIN, values.as_slice());

            let data_page = pb.consume();
            pages.push(vec![data_page]);
        }

        let page_iterator =
            InMemoryPageIterator::new(schema.clone(), column_desc.clone(), pages);

        let mut array_reader =
            ComplexObjectArrayReader::<ByteArrayType, Utf8Converter>::new(
                Box::new(page_iterator),
                column_desc.clone(),
            )
                .unwrap();

        let mut accu_len: usize = 0;

        let array = array_reader.next_batch(values_per_page / 2).unwrap();
        assert_eq!(array.len(), values_per_page / 2);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
        accu_len += array.len();

        // Read next values_per_page values, the first values_per_page/2 ones are from the first column chunk,
        // and the last values_per_page/2 ones are from the second column chunk
        let array = array_reader.next_batch(values_per_page).unwrap();
        assert_eq!(array.len(), values_per_page);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
        let strings = array.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..array.len() {
            if array.is_valid(i) {
                assert_eq!(
                    all_values[i + accu_len].as_ref().unwrap().as_str(),
                    strings.value(i)
                )
            } else {
                assert_eq!(all_values[i + accu_len], None)
            }
        }
        accu_len += array.len();

        // Try to read values_per_page values, however there are only values_per_page/2 values
        let array = array_reader.next_batch(values_per_page).unwrap();
        assert_eq!(array.len(), values_per_page / 2);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
    }

    /// Array reader for test.
    struct InMemoryArrayReader {
        data_type: ArrowType,
        array: ArrayRef,
        def_levels: Option<Vec<i16>>,
        rep_levels: Option<Vec<i16>>,
    }

    impl InMemoryArrayReader {
        pub fn new(
            data_type: ArrowType,
            array: ArrayRef,
            def_levels: Option<Vec<i16>>,
            rep_levels: Option<Vec<i16>>,
        ) -> Self {
            Self {
                data_type,
                array,
                def_levels,
                rep_levels,
            }
        }
    }

    impl ArrayReader for InMemoryArrayReader {
        fn as_any(&self) -> &Any {
            self
        }

        fn get_data_type(&self) -> &ArrowType {
            &self.data_type
        }

        fn next_batch(&mut self, _batch_size: usize) -> Result<ArrayRef> {
            Ok(self.array.clone())
        }

        fn get_def_levels(&self) -> Option<&[i16]> {
            self.def_levels.as_ref().map(|v| v.as_slice())
        }

        fn get_rep_levels(&self) -> Option<&[i16]> {
            self.rep_levels.as_ref().map(|v| v.as_slice())
        }
    }

    #[test]
    fn test_struct_array_reader() {
        let array_1 = Arc::new(PrimitiveArray::<ArrowInt32>::from(vec![1, 2, 3, 4, 5]));
        let array_reader_1 = InMemoryArrayReader::new(
            ArrowType::Int32,
            array_1.clone(),
            Some(vec![0, 1, 2, 3, 1]),
            Some(vec![1, 1, 1, 1, 1]),
        );

        let array_2 = Arc::new(PrimitiveArray::<ArrowInt32>::from(vec![5, 4, 3, 2, 1]));
        let array_reader_2 = InMemoryArrayReader::new(
            ArrowType::Int32,
            array_2.clone(),
            Some(vec![0, 1, 3, 1, 2]),
            Some(vec![1, 1, 1, 1, 1]),
        );

        let struct_type = ArrowType::Struct(vec![
            Field::new("f1", array_1.data_type().clone(), true),
            Field::new("f2", array_2.data_type().clone(), true),
        ]);

        let mut struct_array_reader = StructArrayReader::new(
            struct_type,
            vec![Box::new(array_reader_1), Box::new(array_reader_2)],
            1,
            1,
        );

        let struct_array = struct_array_reader.next_batch(5).unwrap();
        let struct_array = struct_array.as_any().downcast_ref::<StructArray>().unwrap();

        assert_eq!(5, struct_array.len());
        assert_eq!(
            vec![true, false, false, false, false],
            (0..5)
                .map(|idx| struct_array.data_ref().is_null(idx))
                .collect::<Vec<bool>>()
        );
        assert_eq!(
            Some(vec![0, 1, 1, 1, 1].as_slice()),
            struct_array_reader.get_def_levels()
        );
        assert_eq!(
            Some(vec![1, 1, 1, 1, 1].as_slice()),
            struct_array_reader.get_rep_levels()
        );
    }

    #[test]
    fn test_create_array_reader() {
        let file = get_test_file("nulls.snappy.parquet");
        let file_reader = Rc::new(SerializedFileReader::new(file).unwrap());

        let array_reader = build_array_reader(
            file_reader.metadata().file_metadata().schema_descr_ptr(),
            vec![0usize].into_iter(),
            file_reader,
        )
        .unwrap();

        // Create arrow types
        let arrow_type = ArrowType::Struct(vec![Field::new(
            "b_struct",
            ArrowType::Struct(vec![Field::new("b_c_int", ArrowType::Int32, true)]),
            true,
        )]);

        assert_eq!(array_reader.get_data_type(), &arrow_type);
    }
}
