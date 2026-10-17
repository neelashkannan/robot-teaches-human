import React from 'react';
import {C} from '../theme';

// A body drawn from 13 points. "a" = screen-left side, "b" = screen-right side.
// Origin is the pelvis; y grows downward. A figure is about 450 units tall.
export type Pt = [number, number];
export type Pose = {
  head: Pt;
  neck: Pt;
  sa: Pt; sb: Pt; // shoulders
  ea: Pt; eb: Pt; // elbows
  wa: Pt; wb: Pt; // wrists
  ha: Pt; hb: Pt; // hips
  ka: Pt; kb: Pt; // knees
  fa: Pt; fb: Pt; // ankles
};

export const LIMBS = [
  {id: 'ua_a', name: 'Upper arm', from: 'sa', to: 'ea'},
  {id: 'fa_a', name: 'Forearm', from: 'ea', to: 'wa'},
  {id: 'ua_b', name: 'Upper arm', from: 'sb', to: 'eb'},
  {id: 'fa_b', name: 'Forearm', from: 'eb', to: 'wb'},
  {id: 'th_a', name: 'Thigh', from: 'ha', to: 'ka'},
  {id: 'sh_a', name: 'Shin', from: 'ka', to: 'fa'},
  {id: 'th_b', name: 'Thigh', from: 'hb', to: 'kb'},
  {id: 'sh_b', name: 'Shin', from: 'kb', to: 'fb'},
] as const;
export type LimbId = (typeof LIMBS)[number]['id'];

const up = (p: Pose, dy: number): Pose => {
  const out = {...p};
  (['head', 'neck', 'sa', 'sb', 'ea', 'eb', 'wa', 'wb'] as const).forEach((k) => {
    out[k] = [p[k][0], p[k][1] + dy];
  });
  return out;
};

const READY: Pose = {
  head: [0, -228], neck: [0, -192],
  sa: [-50, -180], sb: [50, -180],
  ea: [-60, -104], eb: [60, -104],
  wa: [-64, -32], wb: [64, -32],
  ha: [-28, 0], hb: [28, 0],
  ka: [-31, 96], kb: [31, 96],
  fa: [-32, 192], fb: [32, 192],
};

export const POSES: Record<string, Pose> = {
  ready: READY,
  guard: {
    ...READY,
    ea: [-66, -118], eb: [62, -114],
    wa: [-40, -200], wb: [36, -196],
    ka: [-38, 94], kb: [44, 92],
    fa: [-50, 190], fb: [56, 190],
  },
  horse: {
    ...up(READY, 62),
    ea: [-84, -46], eb: [84, -46],
    wa: [-52, 22], wb: [52, 22],
    ha: [-30, 62], hb: [30, 62],
    ka: [-116, 108], kb: [116, 108],
    fa: [-132, 192], fb: [132, 192],
  },
  crane: {
    ...READY,
    ea: [-122, -206], eb: [122, -206],
    wa: [-192, -242], wb: [192, -242],
    ka: [-26, 96], fa: [-28, 192],
    kb: [62, 34], fb: [48, 112],
  },
  archer: {
    ...up(READY, 14),
    eb: [128, -168], wb: [206, -170],
    ea: [-118, -164], wa: [-40, -160],
    ha: [-30, 14], hb: [30, 14],
    ka: [-82, 100], kb: [86, 102],
    fa: [-126, 192], fb: [128, 192],
  },
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const mixPose = (a: Pose, b: Pose, t: number): Pose => {
  const out = {} as Pose;
  (Object.keys(a) as (keyof Pose)[]).forEach((k) => {
    out[k] = [lerp(a[k][0], b[k][0], t), lerp(a[k][1], b[k][1], t)];
  });
  return out;
};

// Walk a list of [frame, poseName] keys with a smooth blend between neighbours.
export const poseAt = (frame: number, keys: [number, string][], blend = 14): Pose => {
  let pose = POSES[keys[0][1]];
  for (let i = 1; i < keys.length; i++) {
    const [f, name] = keys[i];
    const t = Math.min(1, Math.max(0, (frame - f) / blend));
    const s = t * t * (3 - 2 * t);
    if (frame >= f) pose = mixPose(pose, POSES[name], s);
  }
  return pose;
};

type Props = {
  pose: Pose;
  variant: 'human' | 'robot';
  x: number;
  y: number;
  scale?: number;
  limbColor?: Partial<Record<LimbId, string>>;
  bodyOpacity?: number;
  glow?: string;
};

// Renders inside a parent <svg>.
export const Figure: React.FC<Props> = ({pose, variant, x, y, scale = 1, limbColor = {}, bodyOpacity = 1, glow}) => {
  const robot = variant === 'robot';
  const base = robot ? C.robot : C.skin;
  const limbW = robot ? 22 : 16;
  const p = pose;
  const torso = `M ${p.sa[0]} ${p.sa[1]} L ${p.sb[0]} ${p.sb[1]} L ${p.hb[0]} ${p.hb[1]} L ${p.ha[0]} ${p.ha[1]} Z`;
  return (
    <g transform={`translate(${x} ${y}) scale(${scale})`} opacity={bodyOpacity}>
      {glow ? (
        <ellipse cx={0} cy={200} rx={150} ry={22} fill={glow} opacity={0.35} />
      ) : (
        <ellipse cx={0} cy={200} rx={120} ry={16} fill="#000" opacity={0.35} />
      )}
      <path
        d={torso}
        fill={robot ? C.robotDark : '#33465C'}
        stroke={robot ? C.robot : '#4B6582'}
        strokeWidth={robot ? 8 : 6}
        strokeLinejoin="round"
      />
      {robot && (
        <rect x={-26} y={(p.sa[1] + p.ha[1]) / 2 - 34} width={52} height={40} rx={8} fill={C.robot} opacity={0.9} />
      )}
      <line x1={p.neck[0]} y1={p.neck[1]} x2={(p.sa[0] + p.sb[0]) / 2} y2={p.sa[1]} stroke={base} strokeWidth={14} strokeLinecap="round" />
      {LIMBS.map((l) => {
        const a = p[l.from as keyof Pose];
        const b = p[l.to as keyof Pose];
        const col = limbColor[l.id] ?? base;
        return (
          <g key={l.id}>
            {robot && <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#7D8794" strokeWidth={limbW + 6} strokeLinecap="round" />}
            <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={col} strokeWidth={limbW} strokeLinecap="round" />
          </g>
        );
      })}
      {(['sa', 'sb', 'ea', 'eb', 'ha', 'hb', 'ka', 'kb'] as const).map((k) => (
        <circle key={k} cx={p[k][0]} cy={p[k][1]} r={robot ? 9 : 6} fill={robot ? C.robotDark : '#fff'} stroke={robot ? C.robot : 'none'} strokeWidth={3} />
      ))}
      {robot ? (
        <g transform={`translate(${p.head[0]} ${p.head[1]})`}>
          <rect x={-26} y={-34} width={52} height={64} rx={22} fill={C.robotDark} stroke={C.robot} strokeWidth={6} />
          <ellipse cx={0} cy={-6} rx={14} ry={11} fill={C.robot} opacity={0.85} />
        </g>
      ) : (
        <circle cx={p.head[0]} cy={p.head[1]} r={30} fill={C.skin} />
      )}
    </g>
  );
};
