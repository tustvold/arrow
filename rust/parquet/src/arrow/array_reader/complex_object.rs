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

use std::cmp::max;

use std::marker::PhantomData;

use std::result::Result::Ok;

use std::vec::Vec;

use arrow::array::ArrayRef;

use arrow::datatypes::DataType as ArrowType;

use crate::arrow::converter::Converter;

use crate::arrow::schema::parquet_to_arrow_field;

use crate::column::page::PageIterator;
use crate::column::reader::ColumnReaderImpl;
use crate::data_type::DataType;
use crate::errors::{ParquetError, Result};

use crate::schema::types::ColumnDescPtr;

use crate::arrow::array_reader::ArrayReader;
use std::any::Any;

/// Primitive array readers are leaves of array reader tree. They accept page iterator
/// and read them into primitive arrays.
pub struct ComplexObjectArrayReader<T, C>
where
    T: DataType,
    C: Converter<Vec<Option<T::T>>, ArrayRef> + 'static,
{
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    column_desc: ColumnDescPtr,
    column_reader: Option<ColumnReaderImpl<T>>,
    _parquet_type_marker: PhantomData<T>,
    _converter_marker: PhantomData<C>,
}

impl<T, C> ArrayReader for ComplexObjectArrayReader<T, C>
where
    T: DataType,
    C: Converter<Vec<Option<T::T>>, ArrayRef> + 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef> {
        // Try to initialized column reader
        if self.column_reader.is_none() {
            let init_result = self.next_column_reader()?;
            if !init_result {
                return Err(general_err!("No page left!"));
            }
        }

        assert!(self.column_reader.is_some());

        let mut data_buffer: Vec<T::T> = Vec::with_capacity(batch_size);
        data_buffer.resize_with(batch_size, T::T::default);

        let mut def_levels_buffer = if self.column_desc.max_def_level() > 0 {
            let mut buf: Vec<i16> = Vec::with_capacity(batch_size);
            buf.resize_with(batch_size, || 0);
            Some(buf)
        } else {
            None
        };

        let mut rep_levels_buffer = if self.column_desc.max_rep_level() > 0 {
            let mut buf: Vec<i16> = Vec::with_capacity(batch_size);
            buf.resize_with(batch_size, || 0);
            Some(buf)
        } else {
            None
        };

        let mut num_read = 0;

        while num_read < batch_size {
            let num_to_read = batch_size - num_read;
            let cur_data_buf = &mut data_buffer[num_read..];
            let cur_def_levels_buf =
                def_levels_buffer.as_mut().map(|b| &mut b[num_read..]);
            let cur_rep_levels_buf =
                rep_levels_buffer.as_mut().map(|b| &mut b[num_read..]);
            let (data_read, levels_read) =
                self.column_reader.as_mut().unwrap().read_batch(
                    num_to_read,
                    cur_def_levels_buf,
                    cur_rep_levels_buf,
                    cur_data_buf,
                )?;

            // Fill space
            if levels_read > data_read {
                def_levels_buffer.iter().for_each(|def_levels_buffer| {
                    let (mut level_pos, mut data_pos) = (levels_read, data_read);
                    while level_pos > 0 && data_pos > 0 {
                        if def_levels_buffer[num_read + level_pos - 1]
                            == self.column_desc.max_def_level()
                        {
                            cur_data_buf.swap(level_pos - 1, data_pos - 1);
                            level_pos -= 1;
                            data_pos -= 1;
                        } else {
                            level_pos -= 1;
                        }
                    }
                });
            }


            let values_read = max(levels_read, data_read);
            num_read += values_read;
            // current page exhausted && page iterator exhausted
            if values_read < num_to_read && !self.next_column_reader()? {
                break;
            }
        }

        data_buffer.truncate(num_read);
        def_levels_buffer
            .iter_mut()
            .for_each(|buf| buf.truncate(num_read));
        rep_levels_buffer
            .iter_mut()
            .for_each(|buf| buf.truncate(num_read));

        self.def_levels_buffer = def_levels_buffer;
        self.rep_levels_buffer = rep_levels_buffer;

        let data: Vec<Option<T::T>> = if self.def_levels_buffer.is_some() {
            data_buffer
                .into_iter()
                .zip(self.def_levels_buffer.as_ref().unwrap().iter())
                .map(|(t, def_level)| {
                    if *def_level == self.column_desc.max_def_level() {
                        Some(t)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            data_buffer.into_iter().map(|t| Some(t)).collect()
        };

        C::convert(data)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_ref().map(|t| t.as_slice())
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_ref().map(|t| t.as_slice())
    }
}

impl<T, C> ComplexObjectArrayReader<T, C>
where
    T: DataType,
    C: Converter<Vec<Option<T::T>>, ArrayRef> + 'static,
{
    pub fn new(pages: Box<dyn PageIterator>, column_desc: ColumnDescPtr) -> Result<Self> {
        let data_type = parquet_to_arrow_field(column_desc.as_ref())?
            .data_type()
            .clone();

        Ok(Self {
            data_type,
            pages,
            def_levels_buffer: None,
            rep_levels_buffer: None,
            column_desc,
            column_reader: None,
            _parquet_type_marker: PhantomData,
            _converter_marker: PhantomData,
        })
    }

    fn next_column_reader(&mut self) -> Result<bool> {
        Ok(match self.pages.next() {
            Some(page) => {
                self.column_reader =
                    Some(ColumnReaderImpl::<T>::new(self.column_desc.clone(), page?));
                true
            }
            None => false,
        })
    }
}
