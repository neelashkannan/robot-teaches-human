import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {Figure, LIMBS, mixPose, Pose, POSES} from '../components/Figure';
import {C, FONT} from '../theme';
import {lineStart} from '../voice/voice';

// Each step starts as its voice line does.
const PB = lineStart('tr2') - 4; // eight arrows
const PC = lineStart('tr3') - 4; // direction only
const PD = lineStart('tr4') - 4; // one tip
export const TRICK_FRAMES = PD + 84;

// The visitor's attempt at horse stance: close, except the left forearm sticks out.
const TRY: Pose = {
  ...POSES.horse,
  ea: [-80, -44], wa: [-48, 26],
  eb: [86, -44], wb: [160, -52],
  ka: [-104, 112], fa: [-124, 194],
  kb: [110, 104], fb: [128, 192],
};

// Facing the camera, the figure's left side is on screen-right ("b").
const SIDE: Record<string, string> = {a: 'Right', b: 'Left'};
const limbLabel = (id: string, name: string) => `${SIDE[id.slice(-1)]} ${name.toLowerCase()}`;
const TOL = 30; // scoring.tolerance_deg in config/settings.yaml

const FLOOR = 880;
const R = {x: 600, s: 0.8};
const HU = {x: 1320, s: 1.1};
const figY = (s: number) => FLOOR - 192 * s;

const dirDeg = (p: Pose, a: keyof Pose, b: keyof Pose) => (Math.atan2(p[b][1] - p[a][1], p[b][0] - p[a][0]) * 180) / Math.PI;
const diff = (x: number, y: number) => {
  let d = Math.abs(x - y) % 360;
  return d > 180 ? 360 - d : d;
};

const Arrow: React.FC<{x1: number; y1: number; x2: number; y2: number; color: string; w: number; draw?: number; opacity?: number}> = ({
  x1, y1, x2, y2, color, w, draw = 1, opacity = 1,
}) => {
  const ex = x1 + (x2 - x1) * draw;
  const ey = y1 + (y2 - y1) * draw;
  const a = Math.atan2(y2 - y1, x2 - x1);
  const h = w * 2.4;
  const head = `M ${ex} ${ey} L ${ex - h * Math.cos(a - 0.5)} ${ey - h * Math.sin(a - 0.5)} L ${ex - h * Math.cos(a + 0.5)} ${ey - h * Math.sin(a + 0.5)} Z`;
  return (
    <g opacity={opacity * Math.min(1, draw * 4)}>
      <line x1={x1} y1={y1} x2={ex - Math.cos(a) * h * 0.6} y2={ey - Math.sin(a) * h * 0.6} stroke={color} strokeWidth={w} strokeLinecap="round" />
      <path d={head} fill={color} />
    </g>
  );
};

const Ruler: React.FC<{x: number; top: number; label: string; opacity: number}> = ({x, top, label, opacity}) => (
  <g opacity={opacity}>
    <line x1={x} y1={top} x2={x} y2={FLOOR} stroke={C.muted} strokeWidth={3} strokeDasharray="8 8" />
    <line x1={x - 16} y1={top} x2={x + 16} y2={top} stroke={C.muted} strokeWidth={3} />
    <line x1={x - 16} y1={FLOOR} x2={x + 16} y2={FLOOR} stroke={C.muted} strokeWidth={3} />
    <text x={x} y={top - 20} textAnchor="middle" fill={C.text} style={{fontFamily: FONT.mono, fontWeight: 700, fontSize: 34}}>
      {label}
    </text>
  </g>
);

const Caption: React.FC<{from: number; to: number; label: string; title: React.ReactNode; body: React.ReactNode}> = ({from, to, label, title, body}) => (
  <div style={{position: 'absolute', left: 0, right: 0, top: 62, textAlign: 'center'}}>
    <FadeUp delay={from} out={to}>
      <Label>{label}</Label>
      <Headline size={70} style={{marginTop: 12}}>
        {title}
      </Headline>
      <Body size={30} style={{marginTop: 12}}>
        {body}
      </Body>
    </FadeUp>
  </div>
);

// Styled like the booth app's coaching line.
const Tip: React.FC = () => (
  <div
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 14,
      marginTop: 4,
      padding: '12px 24px',
      borderRadius: 16,
      background: C.panel2,
      border: `2px solid ${C.red}`,
    }}
  >
    <div style={{width: 18, height: 18, borderRadius: 9, background: C.red}} />
    <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 36, color: C.text}}>Left forearm: lower it</div>
  </div>
);

export const Trick: React.FC = () => {
  const frame = useCurrentFrame();

  const robotPose = mixPose(POSES.ready, POSES.horse, ramp(frame, 56, 76));
  const humanPose = mixPose(POSES.ready, TRY, ramp(frame, 70, 92));

  const bodyFade = 1 - 0.78 * ramp(frame, PB, PB + 30) - 0.22 * ramp(frame, PC, PC + 28);
  const rulers = 1 - ramp(frame, PB - 12, PB + 2);
  const morph = ramp(frame, PC, PC + 52);
  const judged = ramp(frame, PC + 46, PC + 62);
  const degrees = ramp(frame, PC + 56, PC + 74);
  const focus = ramp(frame, PD - 4, PD + 10);

  const cols = [510, 810, 1110, 1410];
  const rows = [440, 790];
  const RAD = 96;

  const limbs = LIMBS.map((l, i) => {
    const from = l.from as keyof Pose;
    const to = l.to as keyof Pose;
    const rd = dirDeg(robotPose, from, to);
    const hd = dirDeg(humanPose, from, to);
    const d = diff(rd, hd);
    // Row 1 arms, row 2 legs; within a row: right upper, right lower, left upper, left lower.
    const cx = cols[i % 4];
    const cy = rows[Math.floor(i / 4)];
    return {l, i, from, to, rd, hd, d, ok: d <= TOL, cx, cy};
  });
  const worst = limbs.reduce((a, b) => (b.d > a.d ? b : a));

  const toScreen = (p: Pose, k: keyof Pose, fx: number, s: number): [number, number] => [fx + p[k][0] * s, figY(s) + p[k][1] * s];
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  const arrowFor = (pose: Pose, fx: number, s: number, deg: number, L: (typeof limbs)[number], len: number) => {
    const [ax, ay] = toScreen(pose, L.from, fx, s);
    const [bx, by] = toScreen(pose, L.to, fx, s);
    const rad = (deg * Math.PI) / 180;
    const tx2 = L.cx + Math.cos(rad) * len;
    const ty2 = L.cy + Math.sin(rad) * len;
    return {x1: lerp(ax, L.cx, morph), y1: lerp(ay, L.cy, morph), x2: lerp(bx, tx2, morph), y2: lerp(by, ty2, morph)};
  };

  return (
    <AbsoluteFill>
      <Backdrop />
      <Caption
        from={0}
        to={PB - 6}
        label="How it marks you fairly"
        title="The robot is 1.3 m tall. You might be 1.8 m."
        body="So how can it compare your body with its own?"
      />
      <Caption
        from={PB}
        to={PC - 6}
        label="Step 1"
        title="Both bodies become the same 8 arrows."
        body="Upper arm, forearm, thigh and shin, on each side."
      />
      <Caption
        from={PC}
        to={PD - 6}
        label="Step 2"
        title="Size is ignored. Only direction counts."
        body={
          <>
            Each of your arrows within {TOL}° of the robot&apos;s turns <span style={{color: C.green}}>green</span>.
          </>
        }
      />
      <Caption
        from={PD}
        to={9999}
        label="Step 3"
        title="The biggest miss becomes one plain tip."
        body={<Tip />}
      />

      <svg width={1920} height={1080} style={{position: 'absolute', inset: 0}}>
        <line x1={260} y1={FLOOR + 2} x2={1660} y2={FLOOR + 2} stroke={C.line} strokeWidth={3} opacity={bodyFade} />
        <Ruler x={R.x - 190} top={figY(R.s) - 262 * R.s} label="1.3 m" opacity={rulers} />
        <Ruler x={HU.x + 200} top={figY(HU.s) - 262 * HU.s} label="1.8 m" opacity={rulers} />
        <Figure pose={robotPose} variant="robot" x={R.x} y={figY(R.s)} scale={R.s} bodyOpacity={bodyFade} />
        <Figure pose={humanPose} variant="human" x={HU.x} y={figY(HU.s)} scale={HU.s} bodyOpacity={bodyFade} />
        <text x={R.x} y={FLOOR + 56} textAnchor="middle" fill={C.muted} opacity={bodyFade} style={{fontFamily: FONT.mono, fontWeight: 700, fontSize: 24, letterSpacing: 3}}>
          ROBOT
        </text>
        <text x={HU.x} y={FLOOR + 56} textAnchor="middle" fill={C.muted} opacity={bodyFade} style={{fontFamily: FONT.mono, fontWeight: 700, fontSize: 24, letterSpacing: 3}}>
          YOU
        </text>

        {/* Compass dials the arrows land in */}
        {limbs.map((L) => {
          const dim = L === worst ? 1 : 1 - 0.55 * focus;
          return (
            <g key={`dial-${L.l.id}`} opacity={morph * dim}>
              {L === worst && focus > 0 && (
                <circle cx={L.cx} cy={L.cy} r={RAD + 22 + 8 * Math.sin(frame / 5)} fill="none" stroke={C.red} strokeWidth={4} opacity={focus * 0.7} />
              )}
              <circle cx={L.cx} cy={L.cy} r={RAD + 14} fill={C.panel} stroke={judged > 0 ? (L.ok ? C.green : C.red) : C.line} strokeWidth={judged > 0 ? 4 : 2} strokeOpacity={0.4 + 0.6 * judged} />
              <text x={L.cx} y={L.cy + RAD + 58} textAnchor="middle" fill={C.text} style={{fontFamily: FONT.body, fontWeight: 600, fontSize: 26}}>
                {limbLabel(L.l.id, L.l.name)}
              </text>
              <text x={L.cx} y={L.cy - RAD - 30} textAnchor="middle" opacity={degrees} fill={L.ok ? C.green : C.red} style={{fontFamily: FONT.mono, fontWeight: 700, fontSize: 30}}>
                {`${Math.round(L.d)}° ${L.ok ? '✓' : '✗'}`}
              </text>
            </g>
          );
        })}

        {/* Robot arrows (white) and your arrows (orange, then green or red) */}
        {limbs.map((L) => {
          const draw = ramp(frame, PB + 8 + L.i * 6, PB + 28 + L.i * 6);
          const dim = L === worst ? 1 : 1 - 0.55 * focus;
          const r = arrowFor(robotPose, R.x, R.s, L.rd, L, RAD * 0.86);
          const h = arrowFor(humanPose, HU.x, HU.s, L.hd, L, RAD * 0.86);
          const hc = judged > 0.5 ? (L.ok ? C.green : C.red) : C.orange;
          return (
            <g key={`arrows-${L.l.id}`} opacity={dim}>
              <Arrow {...r} color={C.text} w={10} draw={draw} />
              <Arrow {...h} color={hc} w={10} draw={draw} opacity={0.95} />
            </g>
          );
        })}
      </svg>

    </AbsoluteFill>
  );
};
