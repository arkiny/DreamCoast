//! Action bindings and the per-frame edge detector.

use std::collections::HashMap;
use std::hash::Hash;

use glam::Vec2;

use super::snapshot::InputSnapshot;
use super::source::{BindingError, InputSource};

/// Bindings from game actions to the physical inputs that trigger them.
///
/// `A` is the game's own action enum (`Move`, `Attack`, …); the framework never
/// names actions itself. An action may bind several sources — it is active when
/// **any** of them is.
#[derive(Clone, Debug)]
pub struct ActionMap<A: Copy + Eq + Hash> {
    bindings: HashMap<A, Vec<InputSource>>,
}

impl<A: Copy + Eq + Hash> Default for ActionMap<A> {
    fn default() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }
}

impl<A: Copy + Eq + Hash> ActionMap<A> {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind another source to an action (duplicates are ignored).
    pub fn bind(&mut self, action: A, source: InputSource) {
        let sources = self.bindings.entry(action).or_default();
        if !sources.contains(&source) {
            sources.push(source);
        }
    }

    /// Chainable [`Self::bind`], for building a map in one expression.
    #[must_use]
    pub fn with(mut self, action: A, source: InputSource) -> Self {
        self.bind(action, source);
        self
    }

    /// Bind by source name (`"W"`, `"Mouse1"`, `"WheelUp"`).
    ///
    /// Returns [`BindingError::UnknownSource`] when the name resolves to nothing;
    /// the action is reported as `<action>` since the map cannot name `A`.
    pub fn bind_name(&mut self, action: A, name: &str) -> Result<(), BindingError> {
        let source = InputSource::from_name(name).ok_or_else(|| BindingError::UnknownSource {
            action: "<action>".to_string(),
            source: name.to_string(),
        })?;
        self.bind(action, source);
        Ok(())
    }

    /// Replace every source bound to an action.
    pub fn rebind(&mut self, action: A, sources: Vec<InputSource>) {
        self.bindings.insert(action, sources);
    }

    /// Drop all of an action's bindings.
    pub fn unbind(&mut self, action: A) {
        self.bindings.remove(&action);
    }

    /// The sources bound to an action — empty when unbound.
    pub fn sources(&self, action: A) -> &[InputSource] {
        self.bindings.get(&action).map_or(&[], Vec::as_slice)
    }

    /// Every bound action, in unspecified order.
    pub fn actions(&self) -> impl Iterator<Item = A> + '_ {
        self.bindings.keys().copied()
    }

    /// Every (action, sources) pair, in unspecified order.
    pub fn iter(&self) -> impl Iterator<Item = (A, &[InputSource])> + '_ {
        self.bindings.iter().map(|(a, s)| (*a, s.as_slice()))
    }

    /// Whether any source bound to `action` is active this frame. Unbound actions
    /// are never active.
    pub fn is_active(&self, action: A, snapshot: &InputSnapshot) -> bool {
        self.sources(action).iter().any(|s| s.is_active(snapshot))
    }

    /// The strongest analog magnitude among the action's sources this frame
    /// (`1.0` for a held key, wheel notches for a wheel source, `0.0` if idle).
    pub fn analog(&self, action: A, snapshot: &InputSnapshot) -> f32 {
        self.sources(action)
            .iter()
            .map(|s| s.analog(snapshot))
            .fold(0.0f32, f32::max)
    }

    /// Number of bound actions.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

/// A map plus the two frames of state that edge detection needs.
///
/// This is what gameplay code holds. Every consumer used to hand-roll `*_prev`
/// booleans against the polled platform state; the transitions live here once
/// instead, computed from snapshots so they are testable without a window.
///
/// Feed it one snapshot per frame:
///
/// ```
/// # use dreamcoast_game::input::{ActionMap, ActionState, InputSnapshot, InputSource};
/// #[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// enum Action {
///     Jump,
/// }
///
/// let map = ActionMap::new().with(Action::Jump, InputSource::key("Space").unwrap());
/// let mut state = ActionState::new(map);
///
/// state.update(&InputSnapshot::default().with_key(0x20, true));
/// assert!(state.pressed(Action::Jump) && state.just_pressed(Action::Jump));
///
/// state.update(&InputSnapshot::default().with_key(0x20, true));
/// assert!(state.pressed(Action::Jump) && !state.just_pressed(Action::Jump));
///
/// state.update(&InputSnapshot::default());
/// assert!(!state.pressed(Action::Jump) && state.just_released(Action::Jump));
/// ```
#[derive(Clone, Debug)]
pub struct ActionState<A: Copy + Eq + Hash> {
    map: ActionMap<A>,
    current: HashMap<A, bool>,
    previous: HashMap<A, bool>,
    snapshot: InputSnapshot,
}

impl<A: Copy + Eq + Hash> Default for ActionState<A> {
    fn default() -> Self {
        Self::new(ActionMap::default())
    }
}

impl<A: Copy + Eq + Hash> ActionState<A> {
    /// Wrap a binding map. No action is active until the first [`Self::update`].
    pub fn new(map: ActionMap<A>) -> Self {
        Self {
            map,
            current: HashMap::new(),
            previous: HashMap::new(),
            snapshot: InputSnapshot::default(),
        }
    }

    /// The bindings in use.
    pub fn map(&self) -> &ActionMap<A> {
        &self.map
    }

    /// Mutable bindings, for rebinding at runtime.
    ///
    /// Rebinding is edge-safe in both directions: an action bound between frames
    /// reads as "not held last frame", so holding its key produces a
    /// `just_pressed` on the next update rather than a silent held state; an
    /// action *unbound* while held reports one final `just_released`, so gameplay
    /// that latched on the press (a charge, a held block) is guaranteed the
    /// release event that unlatches it.
    pub fn map_mut(&mut self) -> &mut ActionMap<A> {
        &mut self.map
    }

    /// The frame this state was last updated from.
    pub fn snapshot(&self) -> &InputSnapshot {
        &self.snapshot
    }

    /// Advance one frame: this frame's states become the previous frame's, then
    /// every bound action is re-evaluated against `snapshot`.
    pub fn update(&mut self, snapshot: &InputSnapshot) {
        std::mem::swap(&mut self.current, &mut self.previous);
        self.current.clear();
        for action in self.map.actions() {
            let active = self.map.is_active(action, snapshot);
            self.current.insert(action, active);
        }
        self.snapshot = snapshot.clone();
    }

    /// Whether the action is active this frame.
    pub fn pressed(&self, action: A) -> bool {
        self.current.get(&action).copied().unwrap_or(false)
    }

    /// Whether the action became active this frame.
    pub fn just_pressed(&self, action: A) -> bool {
        self.pressed(action) && !self.was_pressed(action)
    }

    /// Whether the action stopped being active this frame.
    pub fn just_released(&self, action: A) -> bool {
        !self.pressed(action) && self.was_pressed(action)
    }

    /// Whether the action was active on the previous frame.
    pub fn was_pressed(&self, action: A) -> bool {
        self.previous.get(&action).copied().unwrap_or(false)
    }

    /// The action's analog magnitude this frame (see [`ActionMap::analog`]).
    pub fn analog(&self, action: A) -> f32 {
        self.map.analog(action, &self.snapshot)
    }

    /// A one-dimensional axis from two opposing actions: `+1` for `pos`, `-1` for
    /// `neg`, `0` when both or neither are held.
    pub fn axis1d(&self, neg: A, pos: A) -> f32 {
        f32::from(self.pressed(pos)) - f32::from(self.pressed(neg))
    }

    /// A movement vector from four actions, normalized when diagonal so a
    /// diagonal hold is not `√2` times faster than a straight one.
    ///
    /// `+y` is whatever `pos_y` means to the game (this crate has no opinion on
    /// screen or world orientation).
    pub fn axis2d(&self, neg_x: A, pos_x: A, neg_y: A, pos_y: A) -> Vec2 {
        let v = Vec2::new(self.axis1d(neg_x, pos_x), self.axis1d(neg_y, pos_y));
        if v.length_squared() > 1.0 {
            v.normalize()
        } else {
            v
        }
    }

    /// This frame's mouse movement in pixels (raw motion while captured).
    pub fn mouse_delta(&self) -> Vec2 {
        let (dx, dy) = self.snapshot.mouse_delta();
        Vec2::new(dx as f32, dy as f32)
    }

    /// This frame's cursor position in client-area pixels.
    pub fn mouse_position(&self) -> (i32, i32) {
        self.snapshot.mouse_position()
    }

    /// This frame's wheel movement in notches (+up).
    pub fn wheel(&self) -> f32 {
        self.snapshot.wheel()
    }

    /// Whether the pointer is captured (hidden + pinned for mouse look).
    pub fn captured(&self) -> bool {
        self.snapshot.captured()
    }
}

#[cfg(test)]
mod tests {
    use super::super::source::WheelDirection;
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum Action {
        Left,
        Right,
        Forward,
        Back,
        Attack,
        Zoom,
        Unbound,
    }

    fn test_state() -> ActionState<Action> {
        let map = ActionMap::new()
            .with(Action::Left, InputSource::Key(0x41)) // A
            .with(Action::Right, InputSource::Key(0x44)) // D
            .with(Action::Forward, InputSource::Key(0x57)) // W
            .with(Action::Back, InputSource::Key(0x53)) // S
            .with(Action::Attack, InputSource::MouseButton(0))
            .with(Action::Attack, InputSource::Key(0x20)) // Space, second binding
            .with(Action::Zoom, InputSource::Wheel(WheelDirection::Up));
        ActionState::new(map)
    }

    #[test]
    fn edges_across_three_frames() {
        let mut state = test_state();
        let held = InputSnapshot::default().with_key(0x57, true);

        state.update(&held);
        assert!(state.pressed(Action::Forward));
        assert!(state.just_pressed(Action::Forward));
        assert!(!state.just_released(Action::Forward));

        state.update(&held);
        assert!(state.pressed(Action::Forward));
        assert!(
            !state.just_pressed(Action::Forward),
            "held is not a new press"
        );
        assert!(!state.just_released(Action::Forward));

        state.update(&InputSnapshot::default());
        assert!(!state.pressed(Action::Forward));
        assert!(!state.just_pressed(Action::Forward));
        assert!(state.just_released(Action::Forward));

        state.update(&InputSnapshot::default());
        assert!(!state.just_released(Action::Forward), "release fires once");
    }

    #[test]
    fn nothing_is_active_before_the_first_update() {
        let state = test_state();
        assert!(!state.pressed(Action::Forward));
        assert!(!state.just_pressed(Action::Forward));
        assert!(!state.just_released(Action::Forward));
        assert_eq!(
            state.axis2d(Action::Left, Action::Right, Action::Back, Action::Forward),
            Vec2::ZERO
        );
    }

    #[test]
    fn a_tap_lasting_one_frame_reports_both_edges_in_order() {
        let mut state = test_state();
        state.update(&InputSnapshot::default().with_mouse_button(0, true));
        assert!(state.just_pressed(Action::Attack));
        state.update(&InputSnapshot::default());
        assert!(state.just_released(Action::Attack));
    }

    #[test]
    fn any_bound_source_activates_the_action() {
        let mut state = test_state();
        state.update(&InputSnapshot::default().with_key(0x20, true));
        assert!(state.pressed(Action::Attack), "space is the second binding");

        // Both sources at once is still one press, and releasing one keeps it held.
        state.update(
            &InputSnapshot::default()
                .with_key(0x20, true)
                .with_mouse_button(0, true),
        );
        assert!(state.pressed(Action::Attack));
        assert!(!state.just_pressed(Action::Attack));
        state.update(&InputSnapshot::default().with_mouse_button(0, true));
        assert!(state.pressed(Action::Attack));
        assert!(!state.just_released(Action::Attack));
    }

    #[test]
    fn unbound_actions_are_inert() {
        let mut state = test_state();
        assert!(state.map().sources(Action::Unbound).is_empty());
        // Every key in the table down at once.
        let mut all = InputSnapshot::default();
        for vk in 0..256u16 {
            all.set_key(vk, true);
        }
        state.update(&all);
        assert!(!state.pressed(Action::Unbound));
        assert!(!state.just_pressed(Action::Unbound));
        assert!(!state.just_released(Action::Unbound));
        assert_eq!(state.analog(Action::Unbound), 0.0);
    }

    #[test]
    fn axis2d_is_unit_length_on_the_diagonal() {
        let mut state = test_state();
        state.update(
            &InputSnapshot::default()
                .with_key(0x57, true)
                .with_key(0x44, true),
        );
        let v = state.axis2d(Action::Left, Action::Right, Action::Back, Action::Forward);
        assert!((v.length() - 1.0).abs() < 1e-6, "{v:?}");
        assert!(
            (v.x - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "{v:?}"
        );
        assert!(
            (v.y - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6,
            "{v:?}"
        );
    }

    #[test]
    fn axis2d_straight_and_opposing_cases() {
        let mut state = test_state();

        state.update(&InputSnapshot::default().with_key(0x57, true));
        assert_eq!(
            state.axis2d(Action::Left, Action::Right, Action::Back, Action::Forward),
            Vec2::new(0.0, 1.0),
            "a straight hold keeps full magnitude"
        );

        state.update(&InputSnapshot::default().with_key(0x41, true));
        assert_eq!(
            state.axis2d(Action::Left, Action::Right, Action::Back, Action::Forward),
            Vec2::new(-1.0, 0.0)
        );

        state.update(
            &InputSnapshot::default()
                .with_key(0x41, true)
                .with_key(0x44, true),
        );
        assert_eq!(
            state.axis2d(Action::Left, Action::Right, Action::Back, Action::Forward),
            Vec2::ZERO,
            "opposing keys cancel"
        );

        state.update(&InputSnapshot::default());
        assert_eq!(state.axis1d(Action::Left, Action::Right), 0.0);
    }

    #[test]
    fn wheel_is_a_momentary_source_with_an_analog_magnitude() {
        let mut state = test_state();

        state.update(&InputSnapshot::default().with_wheel(2.0));
        assert!(state.pressed(Action::Zoom));
        assert!(state.just_pressed(Action::Zoom));
        assert_eq!(state.analog(Action::Zoom), 2.0);
        assert_eq!(state.wheel(), 2.0);

        // Wheel state does not persist: the next frame with no scroll releases it.
        state.update(&InputSnapshot::default());
        assert!(!state.pressed(Action::Zoom));
        assert!(state.just_released(Action::Zoom));
        assert_eq!(state.analog(Action::Zoom), 0.0);

        // The opposite direction never activates an up-bound action.
        state.update(&InputSnapshot::default().with_wheel(-3.0));
        assert!(!state.pressed(Action::Zoom));
    }

    #[test]
    fn passthroughs_track_the_latest_snapshot() {
        let mut state = test_state();
        state.update(
            &InputSnapshot::default()
                .with_mouse_delta(-3, 8)
                .with_mouse_position(100, 50)
                .with_captured(true),
        );
        assert_eq!(state.mouse_delta(), Vec2::new(-3.0, 8.0));
        assert_eq!(state.mouse_position(), (100, 50));
        assert!(state.captured());
        assert_eq!(state.snapshot().mouse_delta(), (-3, 8));
    }

    #[test]
    fn rebinding_at_runtime_takes_effect_next_frame() {
        let mut state = test_state();
        let snap = InputSnapshot::default().with_key(0x51, true); // Q

        state.update(&snap);
        assert!(!state.pressed(Action::Attack));

        state
            .map_mut()
            .rebind(Action::Attack, vec![InputSource::Key(0x51)]);
        state.update(&snap);
        assert!(state.pressed(Action::Attack));
        assert!(
            state.just_pressed(Action::Attack),
            "a fresh binding reads as a new press"
        );

        // Unbinding a held action must not strand a latched press: the release
        // edge fires once, then the action goes quiet.
        state.map_mut().unbind(Action::Attack);
        state.update(&snap);
        assert!(!state.pressed(Action::Attack));
        assert!(state.just_released(Action::Attack));
        state.update(&snap);
        assert!(!state.just_released(Action::Attack));
        assert!(!state.pressed(Action::Attack));
    }

    #[test]
    fn bind_helpers() {
        let mut map: ActionMap<Action> = ActionMap::new();
        assert!(map.is_empty());
        map.bind_name(Action::Forward, "W").unwrap();
        map.bind_name(Action::Forward, "Up").unwrap();
        map.bind_name(Action::Forward, "W").unwrap(); // duplicate, ignored
        assert_eq!(
            map.sources(Action::Forward),
            &[InputSource::Key(0x57), InputSource::Key(0x26)]
        );
        assert_eq!(map.len(), 1);
        assert!(map.bind_name(Action::Attack, "Frobnicate").is_err());
        assert!(map.sources(Action::Attack).is_empty());
    }
}
