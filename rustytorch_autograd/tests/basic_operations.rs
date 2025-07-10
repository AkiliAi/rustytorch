// //! Tests pour les opérations de base
// //!
// //! Tests exhaustifs pour addition, soustraction, multiplication, division, etc.
//
// use rustytorch_autograd::{Variable, enable_grad};
//
// use crate::gradient_validation::{gradient_check, DEFAULT_TOLERANCE};
//
// #[test]
// fn test_addition_simple() {
//     let x = Variable::variable_with_grad(&[2.0], vec![1]);
//     let y = Variable::variable_with_grad(&[3.0], vec![1]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| inputs[0].add(&inputs[1]),
//         DEFAULT_TOLERANCE,
//         "Simple Addition",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_addition_vectors() {
//     let x = Variable::variable_with_grad(&[1.0, 2.0, 3.0], vec![3]);
//     let y = Variable::variable_with_grad(&[0.5, 1.5, 2.5], vec![3]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| inputs[0].add(&inputs[1]),
//         DEFAULT_TOLERANCE,
//         "Vector Addition",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_subtraction() {
//     let x = Variable::variable_with_grad(&[5.0, 4.0], vec![2]);
//     let y = Variable::variable_with_grad(&[2.0, 1.0], vec![2]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| inputs[0].sub(&inputs[1]),
//         DEFAULT_TOLERANCE,
//         "Subtraction (x - y)",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_multiplication_elementwise() {
//     let x = Variable::variable_with_grad(&[2.0, 3.0], vec![2]);
//     let y = Variable::variable_with_grad(&[1.5, 0.5], vec![2]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| inputs[0].mul(&inputs[1]),
//         DEFAULT_TOLERANCE,
//         "Element-wise Multiplication",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_division() {
//     let x = Variable::variable_with_grad(&[6.0, 8.0], vec![2]);
//     let y = Variable::variable_with_grad(&[2.0, 4.0], vec![2]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| inputs[0].div(&inputs[1]),
//         DEFAULT_TOLERANCE,
//         "Division (x / y)",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_power_operation() {
//     let x = Variable::variable_with_grad(&[2.0, 1.5], vec![2]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].pow(2.5),
//         DEFAULT_TOLERANCE,
//         "Power x^2.5",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_chained_operations() {
//     // f(x, y) = (x + y) * (x - y) = x² - y²
//     let x = Variable::variable_with_grad(&[3.0], vec![1]);
//     let y = Variable::variable_with_grad(&[2.0], vec![1]);
//
//     let result = gradient_check(
//         &[x, y],
//         |inputs| {
//             let sum = inputs[0].add(&inputs[1]);
//             let diff = inputs[0].sub(&inputs[1]);
//             sum.mul(&diff)
//         },
//         DEFAULT_TOLERANCE,
//         "Chained: (x + y) * (x - y)",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_complex_polynomial() {
//     // f(x) = 2x³ - 3x² + x - 1
//     let x = Variable::variable_with_grad(&[1.5], vec![1]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| {
//             let x = &inputs[0];
//             let x2 = x.mul(x);
//             let x3 = x2.mul(x);
//
//             let two = Variable::variable_with_grad(&[2.0], vec![1]);
//             let three = Variable::variable_with_grad(&[3.0], vec![1]);
//             let one = Variable::variable_with_grad(&[1.0], vec![1]);
//
//             // 2x³ - 3x² + x - 1
//             let term1 = two.mul(&x3);
//             let term2 = three.mul(&x2);
//             let term3 = x.clone();
//
//             term1.sub(&term2).add(&term3).sub(&one)
//         },
//         DEFAULT_TOLERANCE,
//         "Polynomial: 2x³ - 3x² + x - 1",
//     );
//
//     assert!(result.passed);
// }