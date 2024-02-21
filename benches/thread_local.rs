use criterion::{black_box, BatchSize};

use thread_local::ThreadLocal;

fn main() {
    let mut c = criterion::Criterion::default().configure_from_args();

    c.bench_function("get", |b| {
        let local: ThreadLocal<Box<i32>, ()> = ThreadLocal::new();
        local.get_or(|_| Box::new(6), |_| {});
        b.iter(|| {
            black_box(local.get());
        });
    });

    c.bench_function("insert", |b| {
        b.iter_batched_ref(
            ThreadLocal::<i32, ()>::new,
            |local| {
                black_box(local.get_or(|_| 7, |_| {}));
            },
            BatchSize::SmallInput,
        )
    });
}
