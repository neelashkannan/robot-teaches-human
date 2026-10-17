import React from 'react';
import {AbsoluteFill} from 'remotion';
import {Backdrop, FadeUp, Headline, Label} from '../components/Anim';
import {C, FONT} from '../theme';

export const STACK_FRAMES = 240;

// Each card: what it is, and why we picked it (the rubric asks for both).
const CARDS = [
  {
    tag: 'Robot',
    name: 'MuJoCo + open robot models',
    why: 'No robot access at the venue, so we run the real Unitree G1 model in a physics simulator.',
  },
  {
    tag: 'Eyes',
    name: 'MediaPipe Pose',
    why: 'Finds 33 body points in an ordinary webcam image, on the laptop, in about 22 ms.',
  },
  {
    tag: 'Our own ML',
    name: 'Stance recogniser',
    why: 'Pose embeddings + nearest neighbours: knows which stance you are in, every frame, offline.',
  },
  {
    tag: 'AI sensei',
    name: 'Nebius Token Factory',
    why: 'Qwen3-235B picks the next stance and says one line (median 0.8 s). Embeddings turn “something for balance” into a stance.',
  },
  {
    tag: 'Never stalls',
    name: 'Offline fallback',
    why: 'Nebius is called once per move, never per frame. No key or no Wi-Fi? A rule-based sensei takes over.',
  },
  {
    tag: 'AI tools, disclosed',
    name: 'Claude Code',
    why: 'Helped us write the code and this Remotion video. The voice (OpenAI gpt-audio) and music (Google Lyria) are AI too.',
  },
];

export const Stack: React.FC = () => (
  <AbsoluteFill>
    <Backdrop />
    <div style={{position: 'absolute', left: 110, top: 74}}>
      <FadeUp>
        <Label>Tech stack</Label>
      </FadeUp>
      <FadeUp delay={4}>
        <Headline size={68} style={{marginTop: 12}}>
          Each part earns its place.
        </Headline>
      </FadeUp>
    </div>
    <div
      style={{
        position: 'absolute',
        left: 110,
        right: 110,
        top: 272,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 30,
      }}
    >
      {CARDS.map((c, i) => (
        <FadeUp key={i} delay={16 + i * 16} dy={40}>
          <div
            style={{
              height: 342,
              boxSizing: 'border-box',
              borderRadius: 24,
              padding: '32px 34px',
              background: C.panel,
              border: `2px solid ${i === 5 ? C.orange + '88' : C.line}`,
            }}
          >
            <Label size={19} color={i === 5 ? C.orange : C.muted}>
              {c.tag}
            </Label>
            <div style={{fontFamily: FONT.display, fontWeight: 800, fontSize: 40, color: C.text, marginTop: 14, lineHeight: 1.08}}>{c.name}</div>
            <div style={{fontFamily: FONT.body, fontWeight: 500, fontSize: 26, color: C.muted, marginTop: 16, lineHeight: 1.38}}>{c.why}</div>
          </div>
        </FadeUp>
      ))}
    </div>
  </AbsoluteFill>
);
