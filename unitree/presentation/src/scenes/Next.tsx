import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {Figure, LIMBS, mixPose, Pose, POSES, Pt} from '../components/Figure';
import {C, FONT} from '../theme';
import {lineEnd, lineStart} from '../voice/voice';

// Concept, not built: on a real robot, a mistake brings the teacher over to fix your pose by hand.
const T0 = lineStart('next2'); // "You get it wrong, and it walks over..."
const WALK = [T0 - 6, T0 + 54];
const REACH = [WALK[1], WALK[1] + 24];
const GUIDE = [REACH[1] + 2, REACH[1] + 48];
const LET_GO = [GUIDE[1] + 6, GUIDE[1] + 28];
export const NEXT_FRAMES = Math.max(lineEnd('next3') + 30, LET_GO[1] + 40);

const FLOOR = 900;
const HX = 820; // the visitor
const HS = 1;
const RS = 0.9; // the robot: a bit shorter
const RX = {start: 1780, stop: 1120, back: 1180};
const ZOOM = 1.3; // the stage is drawn at 1x, then scaled up around the floor centre

// Archer pose with the left arm drooping (wrong), and the real archer arm (right).
const WRONG: Pose = {...POSES.archer, eb: [110, -120], wb: [160, -60]};
const RIGHT: Pose = POSES.archer;

const L1 = Math.hypot(10, 76); // robot upper arm (READY shoulder → elbow)
const L2 = Math.hypot(4, 72); // robot forearm

// Two-link reach from shoulder S to target T, elbow pointing down.
const reach = (S: Pt, T: Pt): {E: Pt; W: Pt} => {
  const dx = T[0] - S[0];
  const dy = T[1] - S[1];
  const dist = Math.hypot(dx, dy);
  const d = Math.min(dist, L1 + L2 - 0.5);
  const a = Math.atan2(dy, dx);
  const b = Math.acos(Math.max(-1, Math.min(1, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))));
  const e1: Pt = [S[0] + L1 * Math.cos(a + b), S[1] + L1 * Math.sin(a + b)];
  const e2: Pt = [S[0] + L1 * Math.cos(a - b), S[1] + L1 * Math.sin(a - b)];
  return {E: e1[1] > e2[1] ? e1 : e2, W: [S[0] + (dx / dist) * d, S[1] + (dy / dist) * d]};
};

const dir = (p: Pose, a: keyof Pose, b: keyof Pose) => Math.atan2(p[b][1] - p[a][1], p[b][0] - p[a][0]);
const off = (x: number, y: number) => {
  const d = (Math.abs(x - y) * 180) / Math.PI;
  return d > 180 ? 360 - d : d;
};

export const Next: React.FC = () => {
  const frame = useCurrentFrame();

  // The visitor's arm goes from drooping to level while the robot holds the wrist.
  const fix = ramp(frame, GUIDE[0], GUIDE[1]);
  const human = mixPose(WRONG, RIGHT, fix);
  const limbColor: Record<string, string> = {};
  LIMBS.forEach((l) => {
    const d = off(dir(human, l.from as keyof Pose, l.to as keyof Pose), dir(RIGHT, l.from as keyof Pose, l.to as keyof Pose));
    limbColor[l.id] = d <= 12 ? C.green : C.red;
  });
  const fixed = Object.values(limbColor).every((c) => c === C.green);

  // Robot: walks in, reaches, guides, lets go and steps back.
  const walk = ramp(frame, WALK[0], WALK[1]);
  const back = ramp(frame, LET_GO[0], LET_GO[1]);
  const rx = RX.start + (RX.stop - RX.start) * walk + (RX.back - RX.stop) * back;
  const ry = FLOOR - 192 * RS;
  const stepping = Math.sin(Math.PI * walk) + 0.6 * Math.sin(Math.PI * back);
  const ph = frame * 0.42;
  const s = Math.sin(ph) * Math.min(1, stepping * 1.6);
  const robot: Pose = {
    ...POSES.ready,
    ka: [-31 + 16 * s, 96 - 8 * Math.max(0, s)],
    fa: [-32 + 34 * s, 192 - 14 * Math.max(0, s)],
    kb: [31 - 16 * s, 96 - 8 * Math.max(0, -s)],
    fb: [32 - 34 * s, 192 - 14 * Math.max(0, -s)],
  };
  // Its screen-left hand holds the visitor's left wrist.
  const wristScreen: Pt = [HX + human.wb[0] * HS, FLOOR - 192 * HS + human.wb[1] * HS];
  const target: Pt = [(wristScreen[0] - rx) / RS, (wristScreen[1] - ry) / RS];
  const hold = ramp(frame, REACH[0], REACH[1]) * (1 - ramp(frame, LET_GO[0], LET_GO[1] - 6));
  const {E, W} = reach(robot.sa, target);
  robot.ea = [robot.ea[0] + (E[0] - robot.ea[0]) * hold, robot.ea[1] + (E[1] - robot.ea[1]) * hold];
  robot.wa = [robot.wa[0] + (W[0] - robot.wa[0]) * hold, robot.wa[1] + (W[1] - robot.wa[1]) * hold];
  const bob = -5 * Math.abs(Math.sin(ph)) * Math.min(1, stepping * 1.6);

  const praise = ramp(frame, GUIDE[1] - 2, GUIDE[1] + 10);

  return (
    <AbsoluteFill>
      <Backdrop glow={C.orange} />
      <div style={{position: 'absolute', left: 0, right: 0, top: 64, textAlign: 'center'}}>
        <FadeUp>
          <Label>What&apos;s next · on a real robot</Label>
        </FadeUp>
        <FadeUp delay={6}>
          <Headline size={78} style={{marginTop: 14}}>
            Get it wrong, and the robot <span style={{color: C.orange}}>comes to you.</span>
          </Headline>
        </FadeUp>
        <FadeUp delay={T0}>
          <Body size={31} style={{marginTop: 14}}>
            It walks over and guides your arm into place, like a real sensei.
          </Body>
        </FadeUp>
      </div>

      <svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
        <g transform={`translate(960 ${FLOOR}) scale(${ZOOM}) translate(-960 -${FLOOR})`}>
        <ellipse cx={980} cy={FLOOR + 4} rx={720} ry={26} fill={C.line} opacity={0.5} />
        <Figure pose={human} variant="human" x={HX} y={FLOOR - 192 * HS} scale={HS} limbColor={limbColor} glow={fixed ? C.green : undefined} />
        <g transform={`translate(0 ${bob})`}>
          <Figure pose={robot} variant="robot" x={rx} y={ry} scale={RS} />
        </g>
        {hold > 0.6 && (
          <circle cx={wristScreen[0]} cy={wristScreen[1]} r={16 + 3 * Math.sin(frame / 3)} fill="none" stroke={C.orange} strokeWidth={4} opacity={(hold - 0.6) * 2.5} />
        )}
        </g>
      </svg>

      {/* "Perfect." like the booth app's coaching line */}
      <div
        style={{
          position: 'absolute',
          left: 960 + (HX - 960) * ZOOM - 400,
          top: 318,
          opacity: praise,
          transform: `translateY(${(1 - praise) * 12}px)`,
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '10px 20px',
          borderRadius: 14,
          background: C.panel2,
          border: `2px solid ${C.green}`,
        }}
      >
        <div style={{width: 16, height: 16, borderRadius: 8, background: C.green}} />
        <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 32, color: C.text}}>Perfect. Hold it!</div>
      </div>

      <div style={{position: 'absolute', left: 0, right: 0, top: 960, textAlign: 'center'}}>
        <FadeUp delay={20}>
          <Label color={C.muted} size={20}>
            Concept animation · not built yet
          </Label>
        </FadeUp>
      </div>
    </AbsoluteFill>
  );
};
