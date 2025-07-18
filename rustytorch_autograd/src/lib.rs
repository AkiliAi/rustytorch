// rustytorch_autograd/src/lib.rs

use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};
use std::sync::{Arc, Mutex};

// Unique ID for each variable
lazy_static::lazy_static! {
    static ref NEXT_ID: Mutex<usize> = Mutex::new(0);
}

fn get_next_id() -> usize {
    let mut next_id = NEXT_ID.lock().unwrap();
    let id = *next_id;
    *next_id += 1;
    id
}

// The core data structure
pub struct VariableData {
    pub id: usize,
    pub value: f64,
    pub grad: f64,
    grad_fn: Option<Box<dyn Fn(f64)>>,
    pub visited: bool,
}

#[derive(Clone)]
pub struct Variable(Arc<Mutex<VariableData>>);

impl Variable {
    pub fn new(value: f64) -> Self {
        Self(Arc::new(Mutex::new(VariableData {
            id: get_next_id(),
            value,
            grad: 0.0,
            grad_fn: None,
            visited: false,
        })))
    }

    pub fn backward(&self) {
        let mut stack = vec![self.clone()];
        self.lock().grad = 1.0;

        while let Some(var) = stack.pop() {
            if var.lock().visited {
                continue;
            }
            var.lock().visited = true;

            if let Some(grad_fn) = var.lock().grad_fn.as_ref() {
                grad_fn(var.lock().grad);
            }
        }
    }

    pub fn lock(&self) -> std::sync::MutexGuard<VariableData> {
        self.0.lock().unwrap()
    }
}

impl Add for Variable {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let result = Variable::new(self.lock().value + other.lock().value);
        let self_clone = self.clone();
        let other_clone = other.clone();
        result.lock().grad_fn = Some(Box::new(move |grad| {
            self_clone.lock().grad += grad;
            other_clone.lock().grad += grad;
        }));
        result
    }
}

impl Sub for Variable {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let result = Variable::new(self.lock().value - other.lock().value);
        let self_clone = self.clone();
        let other_clone = other.clone();
        result.lock().grad_fn = Some(Box::new(move |grad| {
            self_clone.lock().grad += grad;
            other_clone.lock().grad -= grad;
        }));
        result
    }
}

impl Mul for Variable {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let result = Variable::new(self.lock().value * other.lock().value);
        let self_clone = self.clone();
        let other_clone = other.clone();
        result.lock().grad_fn = Some(Box::new(move |grad| {
            self_clone.lock().grad += other_clone.lock().value * grad;
            other_clone.lock().grad += self_clone.lock().value * grad;
        }));
        result
    }
}
