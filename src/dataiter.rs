use candle_core::{Result, Tensor};

pub struct DataIter<'a> {
    inputs: &'a Tensor,
    outputs: &'a Tensor,
    indexes_in_bytes: Vec<usize>,
}

impl Iterator for DataIter<'_> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(i) = self.indexes_in_bytes.pop() {
            Some(candle_core::error::zip(
                self.inputs.get(i),
                self.outputs.get(i),
            ))
        } else {
            None
        }
    }
}

impl<'a> DataIter<'a> {
    pub fn new_random(inputs: &'a Tensor, outputs: &'a Tensor) -> Result<Self> {
        use rand::rng;
        use rand::seq::SliceRandom;

        let size = inputs.dim(0)?;
        let mut indexes_in_bytes = (0..size).collect::<Vec<_>>();
        indexes_in_bytes.shuffle(&mut rng());
        Ok(Self {
            inputs,
            outputs,
            indexes_in_bytes,
        })
    }
    pub fn new(inputs: &'a Tensor, outputs: &'a Tensor) -> Result<Self> {
        let size = inputs.dim(0)?;
        let indexes_in_bytes = (0..size).collect::<Vec<_>>();
        Ok(Self {
            inputs,
            outputs,
            indexes_in_bytes,
        })
    }
}
