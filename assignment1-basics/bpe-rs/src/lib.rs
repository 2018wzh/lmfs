mod core;
mod io;
mod lexer;
mod train;

use core::{Engine, Model, Rule};
use numpy::IntoPyArray;
use pyo3::{
    exceptions::{PyKeyError, PyOverflowError, PyTypeError, PyValueError},
    prelude::*,
    pybacked::PyBackedStr,
    types::PyList,
};
use std::sync::Arc;

fn decode_utf8(py: Python<'_>, bytes: &[u8]) -> PyResult<Py<PyAny>> {
    let n = isize::try_from(bytes.len())
        .map_err(|_| PyOverflowError::new_err("decoded text is too large"))?;
    // This CPython function applies the required UTF-8 replacement rule.
    unsafe {
        Ok(Bound::<PyAny>::from_owned_ptr_or_err(
            py,
            pyo3::ffi::PyUnicode_DecodeUTF8(bytes.as_ptr().cast(), n, c"replace".as_ptr()),
        )?
        .unbind())
    }
}

#[pyclass(frozen)]
struct RustBPE {
    engine: Arc<Engine>,
    pyints: Vec<Py<PyAny>>,
}

#[pymethods]
impl RustBPE {
    #[new]
    fn new(
        py: Python<'_>,
        vocab: Vec<Vec<u8>>,
        rules: Vec<Rule>,
        specials: Vec<(String, u16)>,
        public: Vec<u64>,
        num_threads: usize,
    ) -> PyResult<Self> {
        let model = Model::new(vocab, rules, specials, public).map_err(PyValueError::new_err)?;
        let pyints = model
            .public
            .iter()
            .map(|&id| id.into_pyobject(py).map(|v| v.into_any().unbind()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            engine: Arc::new(Engine::new(model, num_threads).map_err(PyValueError::new_err)?),
            pyints,
        })
    }

    fn encode(&self, py: Python<'_>, text: PyBackedStr) -> PyResult<Py<PyAny>> {
        let ids = py.detach(|| self.engine.encode(&text));
        let n = ids.len() as isize;
        unsafe {
            let list = Bound::<PyAny>::from_owned_ptr_or_err(py, pyo3::ffi::PyList_New(n))?;
            for (position, &id) in ids.iter().enumerate() {
                let item = self.pyints[id as usize].as_ptr();
                pyo3::ffi::Py_INCREF(item);
                pyo3::ffi::PyList_SET_ITEM(list.as_ptr(), position as isize, item);
            }
            Ok(list.unbind())
        }
    }

    fn encode_numpy<'py>(&self, py: Python<'py>, text: PyBackedStr) -> PyResult<Bound<'py, PyAny>> {
        let mut ids = py.detach(|| self.engine.encode(&text));
        if !self.engine.model.identity {
            for id in &mut ids {
                *id = self.engine.model.public[*id as usize] as u16;
            }
        }
        Ok(ids.into_pyarray(py).into_any())
    }

    fn decode(&self, py: Python<'_>, ids: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
        let model = &self.engine.model;
        let mut bytes = Vec::with_capacity(ids.len().saturating_mul(4));
        for position in 0..ids.len() {
            let id = unsafe {
                let item = pyo3::ffi::PyList_GET_ITEM(ids.as_ptr(), position as isize);
                if pyo3::ffi::PyLong_Check(item) == 0 {
                    return Err(PyTypeError::new_err("token IDs must be integers"));
                }
                let id = pyo3::ffi::PyLong_AsUnsignedLongLong(item);
                if id == u64::MAX && !pyo3::ffi::PyErr_Occurred().is_null() {
                    return Err(PyErr::fetch(py));
                }
                id
            };
            let index = model.index(id).ok_or_else(|| PyKeyError::new_err(id))?;
            bytes.extend_from_slice(&model.arena[model.offsets[index]..model.offsets[index + 1]]);
        }
        decode_utf8(py, &bytes)
    }

    fn stream(&self) -> RustBPEStream {
        RustBPEStream {
            engine: self.engine.clone(),
            pending: String::new(),
            finished: false,
        }
    }

    #[pyo3(signature=(input_path, output_path, block_bytes=16777216))]
    fn encode_file(
        &self,
        py: Python<'_>,
        input_path: &str,
        output_path: &str,
        block_bytes: usize,
    ) -> PyResult<u64> {
        py.detach(|| io::encode_file(&self.engine, input_path, output_path, block_bytes))
            .map_err(PyValueError::new_err)
    }
}

#[pyclass]
struct RustBPEStream {
    engine: Arc<Engine>,
    pending: String,
    finished: bool,
}

#[pymethods]
impl RustBPEStream {
    #[pyo3(signature=(text, final_chunk=false))]
    fn push(&mut self, py: Python<'_>, text: &str, final_chunk: bool) -> PyResult<Vec<u64>> {
        if self.finished {
            return Err(PyValueError::new_err("stream is already finished"));
        }
        self.pending.push_str(text);
        let cut = if final_chunk {
            self.pending.len()
        } else {
            self.engine.model.specials.stable_cut(&self.pending)
        };
        let ids = py.detach(|| self.engine.encode(&self.pending[..cut]));
        self.pending.drain(..cut);
        self.finished = final_chunk;
        Ok(ids
            .into_iter()
            .map(|id| self.engine.model.public[id as usize])
            .collect())
    }
}

#[pyfunction]
fn train_file(
    py: Python<'_>,
    path: &str,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<String> {
    let model = py
        .detach(|| train::train(path, vocab_size, special_tokens))
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&model).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBPE>()?;
    module.add_class::<RustBPEStream>()?;
    module.add_function(wrap_pyfunction!(train_file, module)?)?;
    Ok(())
}
