import React from 'react';
import {AbsoluteFill, interpolate, OffthreadVideo, spring, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {FadeUp, Label, ramp} from '../components/Anim';
import {ARM, G1_JOINTS} from '../footage';
import {C, FONT} from '../theme';
import {lineStart} from '../voice/voice';

const PUNCH = lineStart('hype2'); // "Now... the whole body."
const REVEAL = PUNCH + 42; // the flash, on "whole body": the silhouette lights up as the G1
export const HYPE_FRAMES = REVEAL + 62;

// Bridge from the small teacher to the big one: hand → whole body.
export const Hype: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const punch = spring({frame: frame - PUNCH, fps, config: {damping: 11, mass: 0.6}});
  const lit = ramp(frame, REVEAL, REVEAL + 6);
  const flash = interpolate(frame, [REVEAL, REVEAL + 2, REVEAL + 14], [0, 0.85, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const shake = frame >= REVEAL ? Math.sin(frame * 2.7) * 9 * Math.max(0, 1 - (frame - REVEAL) / 12) : 0;
  const zoom = spring({frame: frame - REVEAL, fps, config: {damping: 16, mass: 0.8}});
  const count = Math.round(interpolate(frame, [REVEAL + 4, REVEAL + 34], [ARM.motors, G1_JOINTS], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'}));
  const sweep = interpolate(frame, [REVEAL, REVEAL + 40], [-30, 130], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});

  return (
    <AbsoluteFill style={{background: C.bg, transform: `translateX(${shake}px)`}}>
      {/* Light that blooms behind the robot at the reveal */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          opacity: lit,
          background: `radial-gradient(ellipse 38% 60% at 72% 55%, ${C.orange}55 0%, ${C.sky}44 45%, transparent 75%)`,
        }}
      />
      {/* The G1 doing its stance tour: a dark silhouette until the flash */}
      <div
        style={{
          position: 'absolute',
          left: 1000,
          top: 70,
          width: 840,
          height: 945,
          overflow: 'hidden',
          transform: `scale(${1.25 - 0.25 * zoom})`,
          filter: `brightness(${0.1 + 0.9 * lit}) saturate(${lit})`,
          maskImage: 'radial-gradient(ellipse 50% 50% at 50% 50%, black 55%, transparent 100%)',
          WebkitMaskImage: 'radial-gradient(ellipse 50% 50% at 50% 50%, black 55%, transparent 100%)',
        }}
      >
        <OffthreadVideo
          src={staticFile('footage/g1_tour.mp4')}
          trimBefore={3 * fps}
          playbackRate={1.6}
          muted
          style={{width: 840, height: 945, objectFit: 'cover', objectPosition: '50% 20%'}}
        />
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background: `linear-gradient(100deg, transparent ${sweep - 12}%, rgba(255,255,255,0.35) ${sweep}%, transparent ${sweep + 12}%)`,
            mixBlendMode: 'screen',
          }}
        />
      </div>

      <div style={{position: 'absolute', left: 130, top: 250, width: 960}}>
        <FadeUp delay={4} dy={16}>
          <Label color={C.green}>Teacher 1 ✓</Label>
          <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 56, color: C.muted, marginTop: 14}}>That was one {ARM.noun}.</div>
        </FadeUp>
        <div
          style={{
            marginTop: 22,
            opacity: Math.min(1, punch * 1.5),
            transform: `scale(${1.35 - 0.35 * punch})`,
            transformOrigin: 'left center',
            fontFamily: FONT.display,
            fontWeight: 800,
            fontSize: 132,
            lineHeight: 0.98,
            letterSpacing: '-0.03em',
            color: C.text,
          }}
        >
          Now, the <span style={{color: C.orange}}>whole body.</span>
        </div>
        <div style={{marginTop: 54, display: 'flex', alignItems: 'baseline', gap: 26, opacity: ramp(frame, REVEAL, REVEAL + 4)}}>
          <div style={{fontFamily: FONT.mono, fontWeight: 700, fontSize: 150, color: C.orange, lineHeight: 1, width: 190}}>{count}</div>
          <div>
            <div style={{fontFamily: FONT.display, fontWeight: 800, fontSize: 52, color: C.text}}>joints</div>
            <div style={{fontFamily: FONT.body, fontWeight: 500, fontSize: 28, color: C.muted}}>
              up from {ARM.motors} motors in the hand
            </div>
          </div>
        </div>
      </div>
      <div style={{position: 'absolute', left: 130, top: 905}}>
        <FadeUp delay={REVEAL + 14} dy={14}>
          <Label size={26}>Teacher 2 · virtual Unitree G1 humanoid · 1.3 m</Label>
        </FadeUp>
      </div>

      <AbsoluteFill style={{background: '#fff', opacity: flash, pointerEvents: 'none'}} />
    </AbsoluteFill>
  );
};
