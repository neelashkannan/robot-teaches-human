import React from 'react';
import {Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, FONT} from '../theme';

export const useIn = (delay = 0, damping = 18) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - delay, fps, config: {damping, mass: 0.7}});
};

// 0 → 1 over [start, end] frames, eased, clamped.
export const ramp = (frame: number, start: number, end: number) =>
  interpolate(frame, [start, end], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
    easing: Easing.bezier(0.33, 0, 0.2, 1),
  });

export const FadeUp: React.FC<{
  delay?: number;
  out?: number; // frame to start fading out (optional)
  dy?: number;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({delay = 0, out, dy = 28, style, children}) => {
  const frame = useCurrentFrame();
  const p = useIn(delay);
  const o = out === undefined ? 1 : 1 - ramp(frame, out, out + 10);
  return (
    <div style={{opacity: p * o, transform: `translateY(${(1 - p) * dy}px)`, ...style}}>
      {children}
    </div>
  );
};

export const Label: React.FC<{children: React.ReactNode; color?: string; size?: number; style?: React.CSSProperties}> = ({
  children,
  color = C.orange,
  size = 22,
  style,
}) => (
  <div
    style={{
      fontFamily: FONT.mono,
      fontWeight: 700,
      fontSize: size,
      letterSpacing: '0.14em',
      textTransform: 'uppercase',
      color,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Headline: React.FC<{children: React.ReactNode; size?: number; style?: React.CSSProperties}> = ({
  children,
  size = 84,
  style,
}) => (
  <div
    style={{
      fontFamily: FONT.display,
      fontWeight: 800,
      fontSize: size,
      lineHeight: 1.02,
      letterSpacing: '-0.02em',
      color: C.text,
      ...style,
    }}
  >
    {children}
  </div>
);

export const Body: React.FC<{children: React.ReactNode; size?: number; color?: string; style?: React.CSSProperties}> = ({
  children,
  size = 34,
  color = C.muted,
  style,
}) => (
  <div style={{fontFamily: FONT.body, fontWeight: 500, fontSize: size, lineHeight: 1.35, color, ...style}}>
    {children}
  </div>
);

// Subtle dojo-floor grid + vignette so flat scenes aren't a dead black.
export const Backdrop: React.FC<{glow?: string}> = ({glow = C.sky}) => {
  const frame = useCurrentFrame();
  const drift = (frame * 0.4) % 80;
  return (
    <div style={{position: 'absolute', inset: 0, background: C.bg, overflow: 'hidden'}}>
      <div
        style={{
          position: 'absolute',
          inset: -80,
          backgroundImage: `linear-gradient(${C.line}55 1px, transparent 1px), linear-gradient(90deg, ${C.line}55 1px, transparent 1px)`,
          backgroundSize: '80px 80px',
          transform: `translateY(${drift}px)`,
          maskImage: 'linear-gradient(to bottom, transparent 0%, black 55%, black 100%)',
          WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 55%, black 100%)',
          opacity: 0.55,
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(ellipse 70% 55% at 50% 0%, ${glow}55 0%, transparent 70%)`,
        }}
      />
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: 'radial-gradient(ellipse 120% 90% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%)',
        }}
      />
    </div>
  );
};
