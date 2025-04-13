//rustytorch_utils/src/lib.rs



pub mod Logging {
    use std::fmt;
    use std::fmt::Display;

    #[derive(Debug,Clone,PartialEq,Eq,PartialOrd,Ord)]
    pub enum Loglevel {
        Debug,
        Info,
        Warning,
        Error,
        Critical,
    }

    impl Display for Loglevel {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result{
            match self{
                Loglevel::Debug => write!(f,"DEBUG"),
                Loglevel::Info => write!(f,"INFO"),
                Loglevel::Warning => write!(f,"WARNING"),
                Loglevel::Error => write!(f,"ERROR"),
                Loglevel::Critical => write!(f,"CRITICAL"),
            }
        }
    }

    /// Configuration du logger
    pub struct Logger {
        level: Loglevel,
    }

    impl Logger{
        pub fn new(level: Loglevel) -> Self{
            Self{level}
        }
        pub fn log(&self,level: Loglevel,message: &str){
            if level >= self.level {
                println!("[{}] {}", level, message);
            }
        }

        pub fn debug(&self,message: &str) {
            self.log(Loglevel::Debug,message);
        }

        pub fn info(&self,message: &str) {
            self.log(Loglevel::Info,message);
        }

        pub fn warning(&self,message: &str) {
            self.log(Loglevel::Warning,message);
        }
        pub fn error(&self,message: &str) {
            self.log(Loglevel::Error,message);
        }

        pub fn critical(&self,message: &str) {
            self.log(Loglevel::Critical,message);
        }
    }

    impl Default for Logger{
        fn default() -> Self {
            Self::new(Loglevel::Info)
        }
    }

}


pub mod profiling {
    use std::time::{Duration, Instant};

    // Simple Time pour messures la durée d'exécution
    pub struct Timer {
        start:Instant,
        name: String,
    }

    impl Timer{
        pub fn new(name: &str) -> Self {
            println!("[TIMER] Starting: {}",name);
            Self{
                start: Instant::now(),
                name: name.to_string(),
            }
        }
        pub fn elapsed(&self) -> Duration{
            self.start.elapsed()
        }

        pub fn reset(&mut self){
            self.start = Instant::now();
        }
    }

    impl Drop for  Timer{
        fn drop(&mut self) {
            let duration = self.elapsed();
            println!("[TIMER] {} took: {:?}",self.name,duration);
        }
    }
}

    // Module pouur les fonctions de benchmar

pub mod benchmark {
    use super::profiling::Timer;
    use std::time::Duration;

    // Exécute une fonction et mesure le temps d'exécution
    pub fn benchmark<F, R>(name: &str, iterations: u32, f: F) -> BenchmarkResult
    where
        F: Fn() -> R,
    {
        let mut times = Vec::with_capacity(iterations as usize);

        for _ in 0..iterations {
            let timer = Timer::new(&format!("{}:iteration", name));
            f();
            times.push(timer.elapsed());
        }

        // Calculer les statistiques
        BenchmarkResult {
            name: name.to_string(),
            iterations,
            times,
        }
    }

    /// Resultat du benchmark
    pub struct BenchmarkResult{
        pub name : String,
        pub iterations:u32,
        pub times:Vec<Duration>,
    }

    impl BenchmarkResult{
        pub fn avg(&self) -> Duration{
            let sum: Duration = self.times.iter().sum();
            sum / self.iterations
        }
        pub fn min(&self) -> Duration{
            *self.times.iter().min().unwrap_or(&Duration::ZERO)
        }
        pub fn max(&self) -> Duration{
            *self.times.iter().max().unwrap_or(&Duration::ZERO)
        }


        // Optionnel: Calculer la médiane
        pub fn median(&self) -> Duration{
            let mut sorted_times = self.times.clone();
            sorted_times.sort();
            let mid = sorted_times.len() / 2;
            if sorted_times.len() % 2 == 0 {
                (sorted_times[mid - 1] + sorted_times[mid]) / 2
            } else {
                sorted_times[mid]
            }
        }

        pub fn print_summary(&self){
            println!("[BENCHMARK] {}: {} iterations",self.name,self.iterations);
            println!("[BENCHMARK] Average time: {:?}",self.avg());
            println!("[BENCHMARK] Min time: {:?}",self.min());
            println!("[BENCHMARK] Max time: {:?}",self.max());
            // println!("[BENCHMARK] Median time: {:?}",self.median());
        }

    }



}