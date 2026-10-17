import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {C, FONT} from '../theme';

export const WHY_FRAMES = 240;

const IconAngle: React.FC<{p: number}> = ({p}) => (
  <svg width={96} height={96} viewBox="0 0 96 96">
    <line x1={16} y1={78} x2={84} y2={78} stroke={C.text} strokeWidth={6} strokeLinecap="round" />
    <line x1={16} y1={78} x2={16 + 64 * Math.cos(-0.9 * p)} y2={78 + 64 * Math.sin(-0.9 * p)} stroke={C.orange} strokeWidth={6} strokeLinecap="round" />
    <path d={`M 50 78 A 34 34 0 0 0 ${16 + 34 * Math.cos(-0.9 * p)} ${78 + 34 * Math.sin(-0.9 * p)}`} fill="none" stroke={C.green} strokeWidth={4} />
  </svg>
);

const IconRepeat: React.FC<{p: number}> = ({p}) => (
  <svg width={96} height={96} viewBox="0 0 96 96">
    <g transform={`rotate(${p * 360} 48 48)`}>
      <path d="M 48 14 A 34 34 0 1 1 15 40" fill="none" stroke={C.orange} strokeWidth={6} strokeLinecap="round" />
      <path d="M 6 30 L 15 42 L 27 33" fill="none" stroke={C.orange} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
    </g>
    <circle cx={48} cy={48} r={8} fill={C.text} />
  </svg>
);

const IconReverse: React.FC<{p: number}> = ({p}) => (
  <svg width={96} height={96} viewBox="0 0 96 96">
    <g transform={`translate(${-8 + p * 8} 0)`}>
      <path d="M 14 32 L 78 32 M 64 18 L 80 32 L 64 46" fill="none" stroke={C.muted} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
    </g>
    <g transform={`translate(${8 - p * 8} 0)`}>
      <path d="M 82 66 L 18 66 M 32 52 L 16 66 L 32 80" fill="none" stroke={C.orange} strokeWidth={6} strokeLinecap="round" strokeLinejoin="round" />
    </g>
  </svg>
);

const REASONS = [
  {
    icon: IconAngle,
    title: 'Exact',
    body: 'A robot knows the precise angle of every joint, so the move it shows is exact, every time.',
  },
  {
    icon: IconRepeat,
    title: 'Patient',
    body: 'It repeats the move as often as you need. It never gets tired and never judges.',
  },
  {
    icon: IconReverse,
    title: 'Same tech, backwards',
    body: 'The body tracking that lets robots copy people now checks how well people copy the robot.',
  },
];

export const Why: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <Backdrop />
      <div style={{position: 'absolute', left: 120, top: 96}}>
        <FadeUp>
          <Label>Why flip it?</Label>
        </FadeUp>
        <FadeUp delay={4}>
          <Headline size={80} style={{marginTop: 14, maxWidth: 1500}}>
            Robots make surprisingly good teachers.
          </Headline>
        </FadeUp>
      </div>
      <div style={{position: 'absolute', left: 120, right: 120, top: 360, display: 'flex', gap: 40}}>
        {REASONS.map((r, i) => {
          const d = 24 + i * 56;
          const Icon = r.icon;
          return (
            <FadeUp key={i} delay={d} dy={50} style={{flex: 1}}>
              <div
                style={{
                  height: 470,
                  boxSizing: 'border-box',
                  borderRadius: 28,
                  padding: '40px 40px',
                  background: C.panel,
                  border: `2px solid ${i === 2 ? C.orange : C.line}`,
                  boxShadow: '0 24px 70px rgba(0,0,0,0.45)',
                }}
              >
                <Icon p={ramp(frame, d, d + 40)} />
                <div style={{fontFamily: FONT.display, fontWeight: 800, fontSize: 50, color: C.text, marginTop: 26}}>{r.title}</div>
                <Body size={29} style={{marginTop: 16}}>
                  {r.body}
                </Body>
              </div>
            </FadeUp>
          );
        })}
      </div>
      <div style={{position: 'absolute', left: 0, right: 0, top: 880, textAlign: 'center'}}>
        <FadeUp delay={188}>
          <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 46, color: C.text}}>
            Robot learning, <span style={{color: C.orange}}>run in reverse</span>.
          </div>
        </FadeUp>
      </div>
    </AbsoluteFill>
  );
};
