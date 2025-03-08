use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{Array, ArrayD};
use uuid::Uuid;


#[derive(Debug)]
pub struct Function {
    pub name: String, 
    pub backward: Box<dyn Fn(&[Rc<RefCell<Tensor>>], &ArrayD<f32>) -> Vec<Option<ArrayD<f32>>>>,
}

#[derive(Debug)]
pub struct Tensor {
    pub id: Uuid,
    pub data: ArrayD<f32>,
    pub grad: Option<ArrayD<f32>>,
    pub requires_grad: bool, 
    pub is_leaf: bool, 
    
}