// //! Tests pour les fonctions d'activation
// //!
// //! Tests exhaustifs pour ReLU, Sigmoid, Tanh et leurs gradients
//
// use rustytorch_autograd::{Variable, enable_grad,};
//
// // use crate::gradient_validation::{gradient_check, DEFAULT_TOLERANCE};
// // use rustytorch_autograd::graph_manager::gradient_check;
//
// #[test]
// fn test_relu_positive_values() {
//     // Test ReLU avec des valeurs positives uniquement
//     let x = Variable::variable_with_grad(&[0.5, 1.0, 2.0], vec![3]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].relu(),
//         DEFAULT_TOLERANCE,
//         "ReLU - Positive values",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_relu_negative_values() {
//     // Test ReLU avec des valeurs négatives uniquement
//     let x = Variable::variable_with_grad(&[-2.0, -1.0, -0.5], vec![3]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].relu(),
//         DEFAULT_TOLERANCE,
//         "ReLU - Negative values",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_relu_mixed_values() {
//     // Test ReLU avec un mélange de valeurs positives et négatives
//     let x = Variable::variable_with_grad(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].relu(),
//         1e-4, // Tolérance un peu plus relâchée pour la discontinuité en 0
//         "ReLU - Mixed values",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_sigmoid_normal_range() {
//     // Test Sigmoid dans la plage normale
//     let x = Variable::variable_with_grad(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].sigmoid(),
//         DEFAULT_TOLERANCE,
//         "Sigmoid - Normal range",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_sigmoid_extreme_values() {
//     // Test Sigmoid avec des valeurs extrêmes (mais pas trop pour éviter overflow)
//     let x = Variable::variable_with_grad(&[-5.0, -3.0, 3.0, 5.0], vec![4]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].sigmoid(),
//         DEFAULT_TOLERANCE,
//         "Sigmoid - Extreme values",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_tanh_normal_range() {
//     // Test Tanh dans la plage normale
//     let x = Variable::variable_with_grad(&[-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].tanh(),
//         DEFAULT_TOLERANCE,
//         "Tanh - Normal range",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_tanh_extreme_values() {
//     // Test Tanh avec des valeurs extrêmes
//     let x = Variable::variable_with_grad(&[-4.0, -2.0, 2.0, 4.0], vec![4]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].tanh(),
//         DEFAULT_TOLERANCE,
//         "Tanh - Extreme values",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_activation_composition() {
//     // Test composition d'activations: sigmoid(tanh(x))
//     let x = Variable::variable_with_grad(&[-1.0, 0.0, 1.0], vec![3]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].tanh().sigmoid(),
//         DEFAULT_TOLERANCE,
//         "Composition: sigmoid(tanh(x))",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_relu_in_network() {
//     // Test ReLU dans un contexte de réseau de neurones simple
//     // f(x) = ReLU(x * w + b)
//     let x = Variable::variable_with_grad(&[0.5], vec![1]);
//     let w = Variable::variable_with_grad(&[2.0], vec![1]);
//     let b = Variable::variable_with_grad(&[-0.5], vec![1]);
//
//     let result = gradient_check(
//         &[x, w, b],
//         |inputs| {
//             let x = &inputs[0];
//             let w = &inputs[1];
//             let b = &inputs[2];
//             x.mul(w).add(b).relu()
//         },
//         DEFAULT_TOLERANCE,
//         "ReLU in simple network: ReLU(x*w + b)",
//     );
//
//     assert!(result.passed);
// }
//
// #[test]
// fn test_activation_chain() {
//     // Test une chaîne d'activations: ReLU(Sigmoid(Tanh(x)))
//     let x = Variable::variable_with_grad(&[0.5], vec![1]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| inputs[0].tanh().sigmoid().relu(),
//         DEFAULT_TOLERANCE,
//         "Activation chain: ReLU(Sigmoid(Tanh(x)))",
//     );
//
//     assert!(result.passed);
// }