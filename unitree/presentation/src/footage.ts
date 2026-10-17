// Everything about the recordings lives here, so swapping footage never touches a scene.
// Times are seconds in the source file. Files go in public/footage/.

export type Clip = {
  from: number; // start, seconds into the source
  to: number; // end, seconds into the source
  // Optional slow zoom: [time into the clip (s), scale, origin x %, origin y %]
  zoom?: [number, number, number, number][];
};

export type Step = {
  at: number; // seconds into the scene's footage when this step lights up
  title: string;
  detail: string;
};

// Screen recording of the booth, 17 Oct 13:49 (2876x1606, re-encoded to 1920x1072).
export const UNITREE = {
  src: 'footage/unitree.mp4',
  aspect: 1920 / 1072,
  clips: [
    // Fighting guard: watch, copy, hold, held.
    {from: 11.8, to: 16.8},
    // Horse stance: red limbs, "Left forearm: lower it", fixed, held.
    {
      from: 18.0,
      to: 31.0,
      zoom: [
        [0, 1, 50, 50],
        [1.4, 1, 50, 50],
        [2.6, 1.24, 0, 86],
        [7.4, 1.24, 0, 86],
        [8.6, 1, 50, 50],
        [13, 1, 50, 50],
      ],
    },
    // Scorecard at the end of the session.
    {from: 65.2, to: 68.7, zoom: [[0, 1, 50, 50], [3.5, 1.06, 50, 40]]},
  ] as Clip[],
  // Seconds are measured on the joined clips (5 s + 13 s + 3.5 s).
  steps: [
    {at: 0, title: 'Robot shows a stance', detail: 'The G1 holds it so you can look.'},
    {at: 1.6, title: 'You copy it', detail: 'A normal webcam tracks your body.'},
    {at: 3.8, title: 'AI sensei speaks', detail: 'Picks the next stance, says one line.'},
    {at: 7.0, title: 'Red = fix that limb', detail: 'It names the limb and the way to move.'},
    {at: 13.0, title: 'All green? Hold it', detail: '3 seconds held = passed.'},
    {at: 18.0, title: 'Scorecard', detail: '4 of 5 stances held.'},
  ] as Step[],
};

// The robot hand demo (BSL numbers), screen recording 17 Oct 14:42 (2876x1106), cropped to the
// webcam + teacher panels and the instruction lines, re-encoded to 1920x896.
export const ARM = {
  src: 'footage/arm.mp4' as string | null,
  aspect: 1920 / 896,
  label: 'Teacher 1 · virtual robot hand',
  title: 'Sign language, finger by finger',
  noun: 'robot hand', // used in the hype bridge: "That was one robot hand."
  motors: 8, // Pollen AmazingHand: 8 servos. The G1 model has 29 joints.
  clips: [
    // ONE held, TWO with "middle: fix", then held.
    {from: 0.2, to: 9.2, zoom: [[0, 1, 50, 50], [3.2, 1, 50, 50], [4.0, 1.35, 30, 40], [5.4, 1.35, 30, 40], [6.2, 1, 50, 50]]},
    // FIVE ("closest this hand can do") held, then SEVEN with two red fingers.
    {from: 12.6, to: 18.4, zoom: [[0, 1, 50, 50], [4.0, 1, 50, 50], [4.8, 1.45, 30, 45], [5.8, 1.45, 30, 45]]},
  ] as Clip[],
  steps: [
    {at: 0, title: 'Robot hand shows a sign', detail: 'The number one, in British Sign Language.'},
    {at: 1.2, title: 'You copy it', detail: 'The webcam finds 21 points on your hand.'},
    {at: 3.8, title: 'Each finger marked', detail: 'Red names the finger to fix.'},
    {at: 6.0, title: 'Hold it 3 seconds', detail: 'Then on to the next sign.'},
    {at: 9.2, title: 'Honest about limits', detail: 'Four fingers? It says “closest this hand can do”.'},
  ] as Step[],
};

export const G1_JOINTS = 29; // actuated joints in the Menagerie g1.xml

export const clipSeconds = (clips: Clip[]) => clips.reduce((s, c) => s + (c.to - c.from), 0);
