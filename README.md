# RustyTorch

[//]: # (<div align="center">)

[//]: # ()
[//]: # (![RustyTorch Logo]&#40;https://via.placeholder.com/150x150.png?text=RustyTorch&#41;)

[//]: # ()
[//]: # (**Une bibliothèque de Machine Learning en Rust inspirée de PyTorch**)

[//]: # ()
[//]: # ([![Crates.io]&#40;https://img.shields.io/crates/v/rustytorch.svg&#41;]&#40;https://crates.io/crates/rustytorch&#41;)

[//]: # ([![Documentation]&#40;https://docs.rs/rustytorch/badge.svg&#41;]&#40;https://docs.rs/rustytorch&#41;)

[//]: # ([![License]&#40;https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg&#41;]&#40;LICENSE&#41;)

[//]: # ([![Rust]&#40;https://github.com/username/rustytorch/workflows/Rust/badge.svg&#41;]&#40;https://github.com/username/rustytorch/actions&#41;)

[//]: # ()
[//]: # (</div>)

## À propos

RustyTorch est une bibliothèque de Machine Learning en Rust qui vise à combiner la familiarité de l'API PyTorch avec les avantages de Rust : performance, sécurité mémoire et concurrence sûre. Ce projet est conçu pour les chercheurs et développeurs qui souhaitent exploiter la puissance de Rust tout en conservant la simplicité et la flexibilité qui a rendu PyTorch si populaire.

## Caractéristiques

- 🔢 **Tenseurs performants** - Opérations tensorielles optimisées avec support SIMD et multithreading
- 🧮 **Différentiation automatique** - Système complet de différentiation automatique pour l'entraînement de modèles
- 🧠 **API de réseau neuronal intuitive** - Création de réseaux neuronaux similaire à PyTorch
- 🚀 **Optimiseurs avancés** - SGD, Adam, AdamW et autres optimiseurs
- 🦀 **Idiomatique Rust** - Tire parti des fonctionnalités de Rust comme le système de types, la gestion d'erreurs et les traits
- 🔍 **Vérification à la compilation** - Vérification des dimensions tensorielle à la compilation quand possible
- 🔌 **Extensible** - Architecture modulaire et facile à étendre

## État du projet

RustyTorch est actuellement en phase de développement actif. Les API peuvent changer pendant que nous travaillons à stabiliser la bibliothèque.

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (Ajoutez RustyTorch à votre projet Cargo:)

[//]: # ()
[//]: # (```toml)

[//]: # ([dependencies])

[//]: # (rustytorch = "0.1.0")

[//]: # (```)

[//]: # ()
[//]: # (## Exemple simple)

[//]: # ()
[//]: # (```rust)

[//]: # (use rustytorch::prelude::*;)

[//]: # ()
[//]: # (fn main&#40;&#41; -> Result<&#40;&#41;, Box<dyn std::error::Error>> {)

[//]: # (    // Création d'un tenseur)

[//]: # (    let x = Tensor::new&#40;vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]&#41;?;)

[//]: # (    )
[//]: # (    // Création d'un modèle simple)

[//]: # (    let model = nn::Sequential::new&#40;&#41;)

[//]: # (        .add&#40;nn::Linear::new&#40;2, 4&#41;&#41;)

[//]: # (        .add&#40;nn::ReLU::new&#40;&#41;&#41;)

[//]: # (        .add&#40;nn::Linear::new&#40;4, 1&#41;&#41;;)

[//]: # (    )
[//]: # (    // Prédiction)

[//]: # (    let y = model.forward&#40;&x&#41;?;)

[//]: # (    println!&#40;"Prédiction: {:?}", y&#41;;)

[//]: # (    )
[//]: # (    Ok&#40;&#40;&#41;&#41;)

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## Exemple d'apprentissage)

[//]: # ()
[//]: # (```rust)

[//]: # (use rustytorch::prelude::*;)

[//]: # ()
[//]: # (fn train_linear_regression&#40;&#41; -> Result<&#40;&#41;, Box<dyn std::error::Error>> {)

[//]: # (    // Données d'apprentissage)

[//]: # (    let x_train = Tensor::new&#40;vec![0.1, 0.2, 0.3, 0.4], vec![4, 1]&#41;?;)

[//]: # (    let y_train = Tensor::new&#40;vec![0.2, 0.4, 0.6, 0.8], vec![4, 1]&#41;?;)

[//]: # (    )
[//]: # (    // Modèle, optimiseur et fonction de perte)

[//]: # (    let mut model = nn::Linear::new&#40;1, 1&#41;;)

[//]: # (    let mut optimizer = optim::SGD::new&#40;model.parameters&#40;&#41;, 0.01&#41;;)

[//]: # (    let loss_fn = loss::MSELoss::new&#40;&#41;;)

[//]: # (    )
[//]: # (    // Boucle d'apprentissage)

[//]: # (    for epoch in 0..100 {)

[//]: # (        // Forward pass)

[//]: # (        let pred = model.forward&#40;&x_train&#41;?;)

[//]: # (        let loss = loss_fn.forward&#40;&pred, &y_train&#41;?;)

[//]: # (        )
[//]: # (        // Backward pass et optimisation)

[//]: # (        optimizer.zero_grad&#40;&#41;;)

[//]: # (        loss.backward&#40;&#41;;)

[//]: # (        optimizer.step&#40;&#41;;)

[//]: # (        )
[//]: # (        if epoch % 10 == 0 {)

[//]: # (            println!&#40;"Epoch {}: Loss = {}", epoch, loss.item&#40;&#41;&#41;;)

[//]: # (        })

[//]: # (    })

[//]: # (    )
[//]: # (    // Test)

[//]: # (    let x_test = Tensor::new&#40;vec![0.5], vec![1, 1]&#41;?;)

[//]: # (    let pred = model.forward&#40;&x_test&#41;?;)

[//]: # (    )
[//]: # (    println!&#40;"Prédiction pour x=0.5: {}", pred.item&#40;&#41;&#41;;)

[//]: # (    )
[//]: # (    Ok&#40;&#40;&#41;&#41;)

[//]: # (})

[//]: # (```)

## Architecture

RustyTorch est organisé en plusieurs crates Cargo pour maximiser la modularité et la maintenabilité :

- **rustytorch** - Crate principal (façade)
- **rustytorch-tensor** - Module de tenseurs
- **rustytorch-autograd** - Différentiation automatique
- **rustytorch-nn** - Couches de réseau neuronal
- **rustytorch-optim** - Optimiseurs
- Et plus encore...

## Roadmap

- [ ] Conception de l'architecture
- [ ] Module de tenseurs de base
- [ ] Autograd (différentiation automatique)
- [ ] Couches neuronales fondamentales
- [ ] Optimiseurs
- [ ] Fonctions de perte
- [ ] Support CUDA
- [ ] Modèles préentraînés
- [ ] Bindings Python

Consultez notre [roadmap détaillée](ROADMAP.md) pour plus d'informations.

## Contribution

Les contributions sont les bienvenues ! Veuillez consulter [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives sur la façon de contribuer au projet.

[//]: # ()
[//]: # (## Comparaison avec d'autres bibliothèques)

[//]: # ()
[//]: # (| Fonctionnalité | RustyTorch | PyTorch | TensorFlow | Burn | Candle |)

[//]: # (|----------------|------------|---------|------------|------|--------|)

[//]: # (| Langage | Rust | Python/C++ | Python/C++ | Rust | Rust |)

[//]: # (| Sécurité mémoire | ✅ | ❌ | ❌ | ✅ | ✅ |)

[//]: # (| Vérification à la compilation | ✅ | ❌ | ❌ | ✅ | ✅ |)

[//]: # (| Différentiation automatique | ✅ | ✅ | ✅ | ✅ | ✅ |)

[//]: # (| Support GPU | 🚧 | ✅ | ✅ | ✅ | ✅ |)

[//]: # (| API familière PyTorch | ✅ | ✅ | ❌ | ❌ | ❌ |)

## Licence

RustyTorch est distribué sous les termes d'une double licence - soit sous la licence [MIT](LICENSE-MIT) ou la licence [Apache 2.0](LICENSE-APACHE) au choix.

## Remerciements

RustyTorch s'inspire de plusieurs projets existants :
- [PyTorch](https://pytorch.org/)
- [Burn](https://burn-rs.github.io/)
- [Candle](https://github.com/huggingface/candle)
- [tch-rs](https://github.com/LaurentMazare/tch-rs)

---

<div align="center">
  <sub>Construit avec ❤️ par la communauté Rust ML</sub>
</div>
