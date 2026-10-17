import React from 'react';
import {AbsoluteFill, Img, staticFile} from 'remotion';
import {Body, FadeUp, Headline, useIn} from '../components/Anim';
import {C} from '../theme';
import {lineEnd, lineStart} from '../voice/voice';

export const CLOSE_FRAMES = lineEnd('cl3') + 30;

export const Close: React.FC = () => {
  const zoom = useIn(0, 40);
  return (
    <AbsoluteFill style={{background: C.bg}}>
      <Img
        src={staticFile('img/dojo_booth.png')}
        style={{
          position: 'absolute',
          right: 0,
          top: 0,
          height: 1080,
          width: 960,
          objectFit: 'cover',
          opacity: 0.9,
          transform: `scale(${1.08 - 0.08 * zoom})`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(90deg, ${C.bg} 0%, ${C.bg} 52%, ${C.bg}00 80%)`,
        }}
      />
      <div style={{position: 'absolute', left: 120, top: 300, width: 1000}}>
        <FadeUp delay={4}>
          <Body size={44} color={C.muted}>
            Robots spent a decade learning from us.
          </Body>
        </FadeUp>
        <FadeUp delay={lineStart('cl2') - 2}>
          <Headline size={96} style={{marginTop: 22}}>
            Now they can <span style={{color: C.orange}}>teach us back.</span>
          </Headline>
        </FadeUp>
        <FadeUp delay={lineStart('cl3') - 2}>
          <div style={{marginTop: 70, height: 4, width: 120, background: C.orange, borderRadius: 2}} />
          <Headline size={48} style={{marginTop: 30}}>
            Robot Teaches Human
          </Headline>
          <Body size={28} style={{marginTop: 10}}>
            Come and try it at our table.
          </Body>
        </FadeUp>
      </div>
    </AbsoluteFill>
  );
};
