import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Backdrop, FadeUp, Headline, Label} from '../components/Anim';
import {C, FONT} from '../theme';

export const LIMITS_FRAMES = 180;

const REAL = [
  'Live webcam tracking, scored every frame',
  'Green / red per limb, with a plain-English fix',
  'AI sensei answering live on Nebius in the recording',
];

const NOT_YET = [
  'No robot access here, so both robots live in a virtual world',
  'Stances are posed by hand, not learned by the robot',
  'A webcam judges depth badly, so we mostly score the front view',
  'Stance recogniser checked on synthetic poses so far',
];

const Row: React.FC<{text: string; ok: boolean; delay: number}> = ({text, ok, delay}) => (
  <FadeUp delay={delay} dy={18}>
    <div style={{display: 'flex', gap: 20, alignItems: 'flex-start', marginBottom: 26}}>
      <div
        style={{
          flex: '0 0 auto',
          width: 40,
          height: 40,
          borderRadius: 20,
          marginTop: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: ok ? C.green : C.orange,
          color: C.bg,
          fontFamily: FONT.mono,
          fontWeight: 700,
          fontSize: 22,
        }}
      >
        {ok ? '✓' : '!'}
      </div>
      <div style={{fontFamily: FONT.body, fontWeight: 500, fontSize: 32, color: C.text, lineHeight: 1.3}}>{text}</div>
    </div>
  </FadeUp>
);

export const Limits: React.FC = () => (
  <AbsoluteFill>
    <Backdrop />
    <div style={{position: 'absolute', left: 120, top: 80}}>
      <FadeUp>
        <Label>Honest limits</Label>
      </FadeUp>
      <FadeUp delay={4}>
        <Headline size={72} style={{marginTop: 12}}>
          What&apos;s real, and what isn&apos;t yet.
        </Headline>
      </FadeUp>
    </div>
    <div style={{position: 'absolute', left: 120, right: 120, top: 300, display: 'flex', gap: 80}}>
      <div style={{flex: 1}}>
        <FadeUp delay={14}>
          <Label color={C.green} size={24} style={{marginBottom: 30}}>
            Real, working today
          </Label>
        </FadeUp>
        {REAL.map((t, i) => (
          <Row key={t} text={t} ok delay={20 + i * 12} />
        ))}
      </div>
      <div style={{flex: 1.15}}>
        <FadeUp delay={60}>
          <Label size={24} style={{marginBottom: 30}}>
            Simplified or not built yet
          </Label>
        </FadeUp>
        {NOT_YET.map((t, i) => (
          <Row key={t} text={t} ok={false} delay={66 + i * 12} />
        ))}
      </div>
    </div>
  </AbsoluteFill>
);
