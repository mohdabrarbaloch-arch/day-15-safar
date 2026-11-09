# Safar - usage

A walkthrough of the five things the app does, in the order you'd actually do
them.

---

## 1. Sign in

Click **Sign in** (top right). The demo credentials are pre-filled:

```
demo@safar.pk / safar-demo-2026
```

You can also create your own account - the form toggles between sign-in and
registration. Passwords must be at least 8 characters and mix letters with at
least one other character type.

---

## 2. Run a simulation

The **Simulate** tab is the main view.

1. Pick a **scenario** from the dropdown. The description underneath tells you
   what that scenario is designed to expose.
2. Pick a **driving policy**:
   - *Pure pursuit* - the baseline. Follows the centreline and brakes for
     whatever is ahead.
   - *Q-learning* - a trained policy. You need to train one first (step 4).
3. Set a **seed**. Runs are deterministic, so the same seed and scenario always
   produce identical frames.
4. Click **Run simulation**.

The replay plays automatically. Use the play/pause button, drag the scrubber to
step through time, and toggle the travelled path on or off.

### Reading the metrics

| Metric | What a good number looks like |
|---|---|
| **Safety score** | 80+ is comfortable. Below 55 means something went wrong. |
| **Min time to collision** | Above 2 s is relaxed. Below 1 s is a near miss. `-` means nothing ever closed on you. |
| **Min gap** | The closest you ever got, in metres. Negative means contact. |
| **Max lateral error** | How far you strayed from the centreline. Above 1.75 m is out of a 3.5 m lane. |
| **Max braking** | Peak deceleration. Above 6 m/s is an emergency stop. |
| **Max jerk** | The metric passengers actually feel. Above 6 m/s is uncomfortable. |

Every metric shows *when* it peaked, not just how big it got. A 0.4 s TTC at
4 km/h and a 0.4 s TTC at 60 km/h are very different events, and the timestamp
is how you tell them apart.

---

## 3. Build your own scenario

**Scenarios** tab -> **New scenario**.

Pick a track shape (straight, gentle curve, S-curve, chicane), a weather
condition, a starting speed and a duration. Optionally add:

- a **lead vehicle** at a given speed - this is what tests car-following;
- a **static obstacle** at a given distance - this is what tests braking.

Weather is not cosmetic. It scales braking authority:

| Weather | Braking authority | Speed ceiling |
|---|---|---|
| Dry | 100% | 30 m/s |
| Wet | 62% | 22 m/s |
| Fog | 90% | 16 m/s |

So the same obstacle at the same distance is a comfortable stop in the dry and a
genuine problem in the wet. That is the point of the wet scenario in the built-in
set.

Your scenarios are private to your account. Built-in ones are read-only.

---

## 4. Train an agent

**Train agent** tab.

Tabular Q-learning over 140 discretised states (7 lateral x 5 heading x 4 speed)
and 9 actions (3 accelerations x 3 steering angles).

1. Name the agent.
2. Pick a track.
3. Set episodes (300 is a good default; 10 is enough to see the curve move).
4. Click **Train**.

You get two things back:

- **The reward curve.** Raw reward per episode in faint green, a 20-episode
  moving average in bold. A curve that climbs and then flattens has converged.
- **The learned policy as a heatmap.** Rows are lateral-error buckets, columns
  are speed buckets, and the colour is the average steering command the agent
  settled on. Green means steer left, red means steer right, grey means straight.

That heatmap is the whole reason this is tabular. You can read the policy. A
neural network would score better and teach you nothing.

Training is seeded, so the same settings always produce the same curve.

---

## 5. Review history

**History** tab lists your runs, newest first. Click **Open** on any row to load
that run's replay and metrics back into the Simulate view.

Runs are append-only - nothing is ever mutated after it is written. That is what
makes the history trustworthy as a record.

---

## A worked example: why the wet-road scenario matters

1. Run **Straight - stalled car, wet road** with the baseline.
2. Note the min TTC and the max braking.
3. Now create a scenario with the same obstacle distance but **dry** weather.
4. Run it and compare.

The dry run stops with room to spare. The wet run needs roughly 60% more
distance for the same speed, because braking authority dropped to 62%. Same
policy, same obstacle, different physics - and the metrics show exactly how much
margin the weather cost you.

That is the kind of question this tool exists to answer.
