import React from 'react';
import {AbsoluteFill, interpolate, OffthreadVideo, Sequence, Series, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Clip, clipSeconds, Step} from '../footage';
import {C, FONT} from '../theme';
import {Backdrop, FadeUp, Headline, Label, ramp} from './Anim';

const ZoomedClip: React.FC<{src: string; clip: Clip; frames: number}> = ({src, clip, frames}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  const keys = clip.zoom ?? [[0, 1, 50, 50]];
  const at = (i: number) =>
    keys.length === 1 ? keys[0][i] : interpolate(t, keys.map((k) => k[0]), keys.map((k) => k[i]), {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  // Short dip at each cut so clip joins don't jump.
  const dip = Math.min(ramp(frame, 0, 6), 1 - ramp(frame, frames - 6, frames));
  return (
    <AbsoluteFill style={{background: '#000'}}>
      <AbsoluteFill style={{opacity: dip, transform: `scale(${at(1)})`, transformOrigin: `${at(2)}% ${at(3)}%`}}>
        <OffthreadVideo
          src={staticFile(src)}
          trimBefore={Math.round(clip.from * fps)}
          muted
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export const ClipReel: React.FC<{src: string | null; clips: Clip[]; width: number; height: number}> = ({src, clips, width, height}) => {
  const {fps} = useVideoConfig();
  return (
    <div
      style={{
        width,
        height,
        position: 'relative',
        overflow: 'hidden',
        borderRadius: 22,
        border: `2px solid ${C.line}`,
        boxShadow: '0 30px 80px rgba(0,0,0,0.55)',
        background: C.panel,
      }}
    >
      {src ? (
        <Series>
          {clips.map((c, i) => {
            const frames = Math.round((c.to - c.from) * fps);
            return (
              <Series.Sequence key={i} durationInFrames={frames}>
                <ZoomedClip src={src} clip={c} frames={frames} />
              </Series.Sequence>
            );
          })}
        </Series>
      ) : (
        <Placeholder />
      )}
    </div>
  );
};

// Shown until the arm recording is dropped into public/footage/.
const Placeholder: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill
      style={{
        alignItems: 'center',
        justifyContent: 'center',
        backgroundImage: `repeating-linear-gradient(45deg, ${C.panel} 0 28px, ${C.panel2} 28px 56px)`,
        backgroundPosition: `${frame}px 0`,
      }}
    >
      <div style={{textAlign: 'center'}}>
        <Label size={26}>Recording goes here</Label>
        <div style={{fontFamily: FONT.mono, fontSize: 30, color: C.text, marginTop: 18}}>public/footage/arm.mp4</div>
        <div style={{fontFamily: FONT.body, fontSize: 24, color: C.muted, marginTop: 12}}>then set ARM.src in src/footage.ts</div>
      </div>
    </AbsoluteFill>
  );
};

export const Steps: React.FC<{steps: Step[]; width: number}> = ({steps, width}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;
  let current = 0;
  steps.forEach((s, i) => {
    if (t >= s.at) current = i;
  });
  return (
    <div style={{width, display: 'flex', flexDirection: 'column', gap: 14}}>
      {steps.map((s, i) => {
        const done = i < current;
        const active = i === current;
        const lit = ramp(frame, s.at * fps, s.at * fps + 10);
        return (
          <div
            key={i}
            style={{
              display: 'flex',
              gap: 18,
              alignItems: 'flex-start',
              padding: '16px 18px',
              borderRadius: 16,
              background: active ? C.panel2 : 'transparent',
              border: `2px solid ${active ? C.orange : 'transparent'}`,
              opacity: active ? 1 : done ? 0.6 : 0.28,
              transform: `translateX(${active ? 0 : 6}px)`,
            }}
          >
            <div
              style={{
                flex: '0 0 auto',
                width: 44,
                height: 44,
                borderRadius: 22,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: FONT.mono,
                fontWeight: 700,
                fontSize: 22,
                color: done ? C.bg : active ? C.bg : C.muted,
                background: done ? C.green : active ? C.orange : C.panel2,
              }}
            >
              {done ? '✓' : i + 1}
            </div>
            <div>
              <div style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 31, color: C.text, lineHeight: 1.1}}>{s.title}</div>
              <div
                style={{
                  fontFamily: FONT.body,
                  fontSize: 21,
                  color: C.muted,
                  marginTop: 6,
                  lineHeight: 1.3,
                  maxHeight: active ? 80 : 0,
                  overflow: 'hidden',
                  opacity: active ? lit : 0,
                }}
              >
                {s.detail}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export const demoFrames = (clips: Clip[], fps: number) => Math.round(clipSeconds(clips) * fps);

// Header, footage on the left, a step list on the right that lights up with the recording.
export const DemoScene: React.FC<{
  label: string;
  title: string;
  src: string | null;
  aspect: number;
  clips: Clip[];
  steps: Step[];
  lead?: number; // frames of header-only intro before the footage starts
}> = ({label, title, src, aspect, clips, steps, lead = 0}) => {
  const vw = 1360;
  const vh = Math.round(vw / aspect);
  return (
    <AbsoluteFill>
      <Backdrop />
      <div style={{position: 'absolute', left: 64, top: 52}}>
        <FadeUp>
          <Label>{label}</Label>
        </FadeUp>
        <FadeUp delay={4}>
          <Headline size={64} style={{marginTop: 10}}>
            {title}
          </Headline>
        </FadeUp>
      </div>
      <div style={{position: 'absolute', left: 64, top: 196}}>
        <FadeUp delay={6} dy={40}>
          <SeqFrom from={lead}>
            <ClipReel src={src} clips={clips} width={vw} height={vh} />
          </SeqFrom>
        </FadeUp>
      </div>
      <div style={{position: 'absolute', left: 1468, top: 196}}>
        <SeqFrom from={lead}>
          <Steps steps={steps} width={400} />
        </SeqFrom>
      </div>
    </AbsoluteFill>
  );
};

// Shift a subtree's clock without the absolute positioning <Sequence> adds by default.
const SeqFrom: React.FC<{from: number; children: React.ReactNode}> = ({from, children}) => (
  <Sequence from={from} layout="none">
    {children}
  </Sequence>
);
