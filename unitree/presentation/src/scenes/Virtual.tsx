import React from 'react';
import {AbsoluteFill, OffthreadVideo, Series, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {C, FONT} from '../theme';
import {lineStart} from '../voice/voice';

export const VIRTUAL_FRAMES = 270;

// reports/physics_tour.mp4 (640x720): the G1 in the MuJoCo dojo on its safety rope.
// The top ~64 px is a status line, cropped off below.
const SIM_CLIPS = [
  {from: 28.4, to: 32.4}, // balancing on one leg (crane)
  {from: 118.8, to: 123.8}, // 400 N shove at ~121 s, the rope catches it, it stands again
];
const PW = 680;
const SRC_SCALE = PW / 640;
const CROP = Math.round(64 * SRC_SCALE);
const PH = Math.round(720 * SRC_SCALE) - CROP;

const SimClip: React.FC<{from: number; frames: number}> = ({from, frames}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dip = Math.min(ramp(frame, 0, 5), 1 - ramp(frame, frames - 5, frames));
  return (
    <AbsoluteFill style={{opacity: dip}}>
      <OffthreadVideo
        src={staticFile('footage/sim.mp4')}
        trimBefore={Math.round(from * fps)}
        muted
        style={{position: 'absolute', top: -CROP, left: 0, width: PW, height: PH + CROP}}
      />
    </AbsoluteFill>
  );
};

export const Virtual: React.FC = () => {
  const {fps} = useVideoConfig();
  const frame = useCurrentFrame();
  const pushAt = Math.round((SIM_CLIPS[0].to - SIM_CLIPS[0].from + 2.2) * fps);
  const built = lineStart('virt2');
  return (
    <AbsoluteFill>
      <Backdrop />
      <div style={{position: 'absolute', left: 120, top: 250, width: 900}}>
        <FadeUp>
          <Label>No robot? No problem.</Label>
        </FadeUp>
        <FadeUp delay={4}>
          <Headline size={66} style={{marginTop: 16, color: C.muted}}>
            We couldn&apos;t get access to a real robot here.
          </Headline>
        </FadeUp>
        <FadeUp delay={built - 3}>
          <Headline size={84} style={{marginTop: 26}}>
            So we built them in a <span style={{color: C.orange}}>virtual world.</span>
          </Headline>
        </FadeUp>
        <FadeUp delay={built + 45}>
          <Body size={31} style={{marginTop: 34, maxWidth: 840}}>
            The real Unitree G1 design in MuJoCo, a physics simulator robotics labs use. Gravity, balance and a shove all
            behave like the real thing.
          </Body>
        </FadeUp>
      </div>
      <div style={{position: 'absolute', left: 1120, top: 180}}>
        <FadeUp delay={10} dy={40}>
          <div
            style={{
              width: PW,
              height: PH,
              position: 'relative',
              overflow: 'hidden',
              borderRadius: 24,
              border: `2px solid ${C.line}`,
              boxShadow: '0 30px 80px rgba(0,0,0,0.55)',
              background: '#000',
            }}
          >
            <Series>
              {SIM_CLIPS.map((c, i) => {
                const frames = Math.round((c.to - c.from) * fps);
                return (
                  <Series.Sequence key={i} durationInFrames={frames}>
                    <SimClip from={c.from} frames={frames} />
                  </Series.Sequence>
                );
              })}
            </Series>
            <div
              style={{
                position: 'absolute',
                left: 18,
                top: 16,
                padding: '6px 12px',
                borderRadius: 8,
                background: 'rgba(0,0,0,0.6)',
                fontFamily: FONT.mono,
                fontWeight: 700,
                fontSize: 18,
                letterSpacing: '0.12em',
                color: C.orange,
              }}
            >
              VIRTUAL DOJO · MUJOCO
            </div>
            <div
              style={{
                position: 'absolute',
                left: 18,
                right: 18,
                bottom: 18,
                padding: '12px 16px',
                borderRadius: 12,
                background: 'rgba(10,14,19,0.82)',
                fontFamily: FONT.body,
                fontWeight: 600,
                fontSize: 24,
                color: C.text,
                opacity: ramp(frame, pushAt, pushAt + 8),
              }}
            >
              A hard shove: the safety rope catches it, and it stands back up.
            </div>
          </div>
        </FadeUp>
      </div>
    </AbsoluteFill>
  );
};
