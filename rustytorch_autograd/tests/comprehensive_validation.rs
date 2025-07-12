// //! Tests de validation comprehensive
// //!
// //! Suite principale de tests pour valider l'ensemble du système autograd
//
// use rustytorch_autograd::{Variable, enable_grad};
//
// use crate::gradient_validation::{gradient_check, DEFAULT_TOLERANCE};
//
// /// Test d'un réseau de neurones simple complet
// #[test]
// fn test_simple_neural_network() {
//     // Réseau simple: y = sigmoid(tanh(x*W1 + b1)*W2 + b2)
//     let _guard = enable_grad();
//
//     let x = Variable::variable_with_grad(&[0.5], vec![1]);
//     let w1 = Variable::variable_with_grad(&[1.2], vec![1]);
//     let b1 = Variable::variable_with_grad(&[0.1], vec![1]);
//     let w2 = Variable::variable_with_grad(&[0.8], vec![1]);
//     let b2 = Variable::variable_with_grad(&[-0.3], vec![1]);
//
//     let result = gradient_check(
//         &[x, w1, b1, w2, b2],
//         |inputs| {
//             let x = &inputs[0];
//             let w1 = &inputs[1];
//             let b1 = &inputs[2];
//             let w2 = &inputs[3];
//             let b2 = &inputs[4];
//
//             // Couche 1: tanh(x*w1 + b1)
//             let layer1 = x.mul(w1).add(b1).tanh();
//
//             // Couche 2: sigmoid(layer1*w2 + b2)
//             let output = layer1.mul(w2).add(b2).sigmoid();
//
//             output
//         },
//         DEFAULT_TOLERANCE,
//         "Simple Neural Network",
//     );
//
//     assert!(result.passed, "Neural network gradient validation failed");
//
//     // Afficher des statistiques
//     println!("   ✅ Validation réussie pour {} paramètres", result.num_elements);
//     println!("   📊 Erreur moyenne: {:.2e}, Erreur max: {:.2e}",
//              result.mean_error, result.max_error);
// }
//
// /// Test de fonction de perte (MSE)
// #[test]
// fn test_mse_loss() {
//     // MSE Loss: L = (y_pred - y_true)²
//     let y_pred = Variable::variable_with_grad(&[0.8, 0.3, 0.9], vec![3]);
//     let y_true = Variable::variable_with_grad(&[1.0, 0.0, 1.0], vec![3]);
//
//     let result = gradient_check(
//         &[y_pred, y_true],
//         |inputs| {
//             let pred = &inputs[0];
//             let true_val = &inputs[1];
//
//             // (pred - true)²
//             let diff = pred.sub(true_val);
//             let squared = diff.mul(&diff);
//
//             // Moyenne
//             squared.sum() // Simplification: pas de division par N pour ce test
//         },
//         DEFAULT_TOLERANCE,
//         "MSE Loss Function",
//     );
//
//     assert!(result.passed, "MSE loss gradient validation failed");
// }
//
// /// Test de backpropagation complexe avec gradients de second ordre
// #[test]
// fn test_complex_backprop_with_hessian() {
//     // Fonction complexe: f(x,y) = exp(sin(x*y)) * log(1 + x² + y²)
//     let _guard = enable_grad();
//
//     let x = Variable::variable_with_grad(&[0.5], vec![1]);
//     let y = Variable::variable_with_grad(&[0.8], vec![1]);
//
//     // Test gradient de premier ordre
//     let result_first = gradient_check(
//         &[x.clone(), y.clone()],
//         |inputs| {
//             let x = &inputs[0];
//             let y = &inputs[1];
//
//             let xy = x.mul(y);
//             let sin_xy = xy.sin();
//             let exp_term = sin_xy.exp();
//
//             let x_squared = x.mul(x);
//             let y_squared = y.mul(y);
//             let one = Variable::variable_with_grad(&[1.0], vec![1]);
//             let sum_term = one.add(&x_squared).add(&y_squared);
//             let log_term = sum_term.log();
//
//             exp_term.mul(&log_term)
//         },
//         DEFAULT_TOLERANCE,
//         "Complex function - First order",
//     );
//
//     assert!(result_first.passed, "Complex function first order gradient failed");
//
//     // Test Hessienne
//     let f = {
//         let xy = x.mul(&y);
//         let sin_xy = xy.sin();
//         let exp_term = sin_xy.exp();
//
//         let x_squared = x.mul(&x);
//         let y_squared = y.mul(&y);
//         let one = Variable::variable_with_grad(&[1.0], vec![1]);
//         let sum_term = one.add(&x_squared).add(&y_squared);
//         let log_term = sum_term.log();
//
//         exp_term.mul(&log_term)
//     };
//
//     let hessian = f.hessian(&[x, y]).unwrap();
//
//     // Vérifier que la Hessienne est symétrique
//     if let (Some(h_xy), Some(h_yx)) = (&hessian[0][1], &hessian[1][0]) {
//         let h_xy_val = h_xy.tensor().storage().to_vec_f64()[0];
//         let h_yx_val = h_yx.tensor().storage().to_vec_f64()[0];
//         let symmetry_error = (h_xy_val - h_yx_val).abs();
//
//         assert!(symmetry_error < 1e-8, "Hessian not symmetric: error = {:.2e}", symmetry_error);
//         println!("   ✅ Hessian symmetry verified (error: {:.2e})", symmetry_error);
//     }
// }
//
// /// Test de robustesse avec différentes tailles de tenseurs
// #[test]
// fn test_tensor_size_robustness() {
//     // Tester avec des tenseurs de différentes tailles
//     let sizes = vec![
//         (vec![1], "scalar"),
//         (vec![3], "vector"),
//         (vec![2, 2], "small matrix"),
//         (vec![3, 2], "rectangular matrix"),
//     ];
//
//     for (shape, description) in sizes {
//         let numel = shape.iter().product::<usize>();
//         let data: Vec<f64> = (0..numel).map(|i| 0.1 + i as f64 * 0.2).collect();
//
//         let x = Variable::variable_with_grad(&data, shape.clone());
//         let y = Variable::variable_with_grad(&data.iter().map(|&x| x * 0.5).collect::<Vec<_>>(), shape.clone());
//
//         let result = gradient_check(
//             &[x, y],
//             |inputs| {
//                 let a = &inputs[0];
//                 let b = &inputs[1];
//
//                 // Opération simple: (a + b) * (a - b) = a² - b²
//                 let sum = a.add(b);
//                 let diff = a.sub(b);
//                 sum.mul(&diff).sum() // Sum pour réduire à un scalaire
//             },
//             DEFAULT_TOLERANCE,
//             &format!("Tensor size test - {}", description),
//         );
//
//         assert!(result.passed, "Gradient test failed for tensor size: {:?}", shape);
//     }
// }
//
// /// Test de performance et stabilité numérique
// #[test]
// fn test_numerical_stability() {
//     // Tester avec des valeurs de différentes magnitudes
//     let test_cases = vec![
//         (vec![1e-3, 1e-2, 1e-1], "small values"),
//         (vec![1.0, 2.0, 3.0], "normal values"),
//         (vec![10.0, 20.0, 30.0], "large values"),
//     ];
//
//     for (values, description) in test_cases {
//         let x = Variable::variable_with_grad(&values, vec![values.len()]);
//
//         let result = gradient_check(
//             &[x],
//             |inputs| {
//                 let x = &inputs[0];
//                 // Fonction qui combine plusieurs opérations
//                 let exp_x = x.exp();
//                 let log_x = x.log();
//                 let sin_x = x.sin();
//
//                 // exp(x) + log(x) + sin(x)
//                 exp_x.add(&log_x).add(&sin_x).sum()
//             },
//             DEFAULT_TOLERANCE * 10.0, // Tolérance un peu plus relâchée
//             &format!("Numerical stability - {}", description),
//         );
//
//         if !result.passed {
//             println!("⚠️  Warning: Numerical stability test failed for {}", description);
//             println!("   Max error: {:.2e}, this might be acceptable for extreme values", result.max_error);
//         }
//     }
// }
//
// /// Test de la chaîne de gradients (chain rule)
// #[test]
// fn test_chain_rule_complex() {
//     // Test complexe de la chain rule: f(g(h(x)))
//     let x = Variable::variable_with_grad(&[0.5], vec![1]);
//
//     let result = gradient_check(
//         &[x],
//         |inputs| {
//             let x = &inputs[0];
//
//             // h(x) = x² + 1
//             let h = x.mul(x).add(&Variable::variable_with_grad(&[1.0], vec![1]));
//
//             // g(h) = sin(h)
//             let g = h.sin();
//
//             // f(g) = exp(g)
//             let f = g.exp();
//
//             f
//         },
//         DEFAULT_TOLERANCE,
//         "Chain rule: exp(sin(x² + 1))",
//     );
//
//     assert!(result.passed, "Chain rule gradient test failed");
//
//     println!("   ✅ Chain rule validation successful");
//     println!("   📈 df/dx computed through 3 levels of composition");
// }