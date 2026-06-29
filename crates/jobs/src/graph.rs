//! A **task graph**: nodes (jobs) connected by dependency edges, executed on a
//! [`JobSystem`] in dependency order. This is the CPU analogue of the render
//! graph's GPU-pass DAG — a node runs only once all its dependencies have
//! finished, and independent nodes run in parallel.
//!
//! Scheduling is data-driven and dynamic: each node carries an atomic count of
//! unmet dependencies; finishing a node decrements its successors and schedules
//! any that reach zero. Initial roots are scheduled in insertion order, and
//! successor lists preserve edge-insertion order, so execution is deterministic
//! given a fixed worker count (per the determinism rule).

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::scheduler::{Job, JobSystem, Scope};

/// Identifies a node within a single [`TaskGraph`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TaskId(usize);

/// A directed acyclic graph of jobs to run with respect to their dependencies.
#[derive(Default)]
pub struct TaskGraph {
    jobs: Vec<Option<Job>>,
    deps: Vec<Vec<usize>>, // deps[i] = nodes that must finish before i
}

impl TaskGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node and return its id. Jobs are `'static` (frame-scoped systems);
    /// use [`JobSystem::scope`] directly when you need borrowed data.
    pub fn add<F: FnOnce() + Send + 'static>(&mut self, f: F) -> TaskId {
        let id = self.jobs.len();
        self.jobs.push(Some(Box::new(f)));
        self.deps.push(Vec::new());
        TaskId(id)
    }

    /// Declare that `task` may run only after `before` completes.
    pub fn depend(&mut self, task: TaskId, before: TaskId) {
        assert_ne!(task.0, before.0, "a task cannot depend on itself");
        self.deps[task.0].push(before.0);
    }

    /// Execute the whole graph on `js`, returning once every node has run.
    /// Panics if the graph contains a cycle (a node whose dependencies never
    /// clear).
    pub fn run(self, js: &JobSystem) {
        let n = self.jobs.len();
        if n == 0 {
            return;
        }

        // Invert deps into successor lists + indegree counts.
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let indegree: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        for (node, befores) in self.deps.iter().enumerate() {
            indegree[node].store(befores.len(), Ordering::Relaxed);
            for &before in befores {
                successors[before].push(node);
            }
        }

        let state = Arc::new(GraphState {
            jobs: self.jobs.into_iter().map(Mutex::new).collect(),
            successors,
            indegree,
        });

        js.scope(|s| {
            for i in 0..n {
                if state.indegree[i].load(Ordering::Relaxed) == 0 {
                    schedule(Arc::clone(&state), i, s);
                }
            }
        });

        // Every node must have run exactly once; a leftover means a cycle.
        for (i, slot) in state.jobs.iter().enumerate() {
            assert!(
                slot.lock().unwrap().is_none(),
                "task graph cycle: node {i} never became runnable"
            );
        }
    }
}

struct GraphState {
    jobs: Vec<Mutex<Option<Job>>>,
    successors: Vec<Vec<usize>>,
    indegree: Vec<AtomicUsize>,
}

/// Spawn node `idx`; when it finishes, release successors that reach indegree 0.
fn schedule<'scope>(state: Arc<GraphState>, idx: usize, s: &Scope<'scope>) {
    s.spawn(move |s| {
        let job = state.jobs[idx]
            .lock()
            .unwrap()
            .take()
            .expect("task graph node runs exactly once");
        job();
        for &succ in &state.successors[idx] {
            // AcqRel so the predecessor's writes happen-before the successor runs.
            if state.indegree[succ].fetch_sub(1, Ordering::AcqRel) == 1 {
                schedule(Arc::clone(&state), succ, s);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn runs_in_dependency_order() {
        let js = JobSystem::new(Some(4));
        let order = Arc::new(Mutex::new(Vec::<&'static str>::new()));

        let mut g = TaskGraph::new();
        let (o1, o2, o3, o4) = (
            Arc::clone(&order),
            Arc::clone(&order),
            Arc::clone(&order),
            Arc::clone(&order),
        );
        // Diamond: a -> {b, c} -> d.
        let a = g.add(move || o1.lock().unwrap().push("a"));
        let b = g.add(move || o2.lock().unwrap().push("b"));
        let c = g.add(move || o3.lock().unwrap().push("c"));
        let d = g.add(move || o4.lock().unwrap().push("d"));
        g.depend(b, a);
        g.depend(c, a);
        g.depend(d, b);
        g.depend(d, c);
        g.run(&js);

        let order = order.lock().unwrap();
        let pos = |name| order.iter().position(|&x| x == name).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn runs_every_node_once_in_a_wide_chain() {
        let js = JobSystem::new(Some(4));
        let count = Arc::new(AtomicUsize::new(0));
        let mut g = TaskGraph::new();
        // A linear chain of 100 nodes.
        let mut prev: Option<TaskId> = None;
        for _ in 0..100 {
            let c = Arc::clone(&count);
            let id = g.add(move || {
                c.fetch_add(1, Ordering::Relaxed);
            });
            if let Some(p) = prev {
                g.depend(id, p);
            }
            prev = Some(id);
        }
        g.run(&js);
        assert_eq!(count.load(Ordering::Relaxed), 100);
    }

    #[test]
    #[should_panic(expected = "cycle")]
    fn detects_cycle() {
        let js = JobSystem::new(Some(2));
        let mut g = TaskGraph::new();
        let a = g.add(|| {});
        let b = g.add(|| {});
        g.depend(a, b);
        g.depend(b, a);
        g.run(&js);
    }
}
