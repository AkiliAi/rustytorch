#[cfg(test)]
mod tests {
    use rustytorch_autograd::Variable;

    #[test]
    fn test_add() {
        let a = Variable::new(2.0);
        let b = Variable::new(3.0);
        let c = a.clone() + b.clone();
        assert_eq!(c.lock().value, 5.0);
        c.backward();
        assert_eq!(a.lock().grad, 1.0);
        assert_eq!(b.lock().grad, 1.0);
    }

    #[test]
    fn test_sub() {
        let a = Variable::new(2.0);
        let b = Variable::new(3.0);
        let c = a.clone() - b.clone();
        assert_eq!(c.lock().value, -1.0);
        c.backward();
        assert_eq!(a.lock().grad, 1.0);
        assert_eq!(b.lock().grad, -1.0);
    }

    #[test]
    fn test_mul() {
        let a = Variable::new(2.0);
        let b = Variable::new(3.0);
        let c = a.clone() * b.clone();
        assert_eq!(c.lock().value, 6.0);
        c.backward();
        assert_eq!(a.lock().grad, 3.0);
        assert_eq!(b.lock().grad, 2.0);
    }
}
