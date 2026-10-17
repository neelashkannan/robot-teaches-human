import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import {Backdrop, Body, FadeUp, Headline, Label, ramp} from '../components/Anim';
import {Figure, LIMBS, Pose, poseAt, POSES} from '../components/Figure';
import {C, FONT} from '../theme';

export const FLIP_FRAMES = 330;
const FLIP_AT = 172; // the moment the arrow turns round

// Real G1 renders from reports/robot_stances.png (7 columns of 320x360, top row unmirrored).
const STILL_INDEX: Record<string, number> = {ready: 0, horse: 1, guard: 3, crane: 4, archer: 6};
const PANEL_W = 560;
const PANEL_H = 525;

const RobotStill: React.FC<{name: string; opacity: number}> = ({name, opacity}) => {
  const s = PANEL_W / 320;
  const i = STILL_INDEX[name];
  return (
    <div style={{position: 'absolute', inset: 0, overflow: 'hidden', opacity}}>
      <Img
        src={staticFile('img/robot_stances.png')}
        style={{
          position: 'absolute',
          width: 2240 * s,
          height: 720 * s,
          left: -i * 320 * s,
          top: -30 * s,
          maxWidth: 'none',
        }}
      />
    </div>
  );
};

// The robot's pose at a frame, as a list of [frame, pose] keys; crossfade between stills.
const RobotPanel: React.FC<{keys: [number, string][]}> = ({keys}) => {
  const frame = useCurrentFrame();
  let idx = 0;
  keys.forEach(([f], i) => {
    if (frame >= f) idx = i;
  });
  const cur = keys[idx];
  const prev = keys[Math.max(0, idx - 1)];
  const t = idx === 0 ? 1 : ramp(frame, cur[0], cur[0] + 8);
  return (
    <>
      {idx > 0 && <RobotStill name={prev[1]} opacity={1} />}
      <RobotStill name={cur[1]} opacity={t} />
    </>
  );
};

const angle = (p: Pose, from: keyof Pose, to: keyof Pose) => Math.atan2(p[to][1] - p[from][1], p[to][0] - p[from][0]);
const diffDeg = (a: number, b: number) => {
  let d = Math.abs(a - b) * (180 / Math.PI);
  if (d > 180) d = 360 - d;
  return d;
};

const Panel: React.FC<{x: number; tag: string; tagColor: string; children: React.ReactNode; border?: string}> = ({
  x,
  tag,
  tagColor,
  children,
  border = C.line,
}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: 356,
      width: PANEL_W,
      height: PANEL_H,
      borderRadius: 26,
      overflow: 'hidden',
      border: `3px solid ${border}`,
      background: C.panel,
      boxShadow: '0 30px 80px rgba(0,0,0,0.5)',
    }}
  >
    {children}
    <div
      style={{
        position: 'absolute',
        left: 18,
        top: 16,
        padding: '6px 12px',
        borderRadius: 8,
        background: 'rgba(0,0,0,0.55)',
        fontFamily: FONT.mono,
        fontWeight: 700,
        fontSize: 18,
        letterSpacing: '0.12em',
        color: tagColor,
      }}
    >
      {tag}
    </div>
  </div>
);

const RoleLabel: React.FC<{x: number; who: string; role: string; flip: number; role2: string}> = ({x, who, role, role2, flip}) => {
  // Flip the role word like a card: first half shows the old role, second half the new one.
  const showNew = flip >= 0.5;
  const rot = showNew ? (1 - flip) * 180 : flip * 180;
  return (
    <div style={{position: 'absolute', left: x, top: 906, width: PANEL_W, textAlign: 'center', perspective: 600}}>
      <span style={{fontFamily: FONT.display, fontWeight: 700, fontSize: 40, color: C.text}}>{who} = </span>
      <span
        style={{
          display: 'inline-block',
          fontFamily: FONT.display,
          fontWeight: 800,
          fontSize: 40,
          color: showNew ? C.orange : C.muted,
          transform: `rotateX(${rot}deg)`,
        }}
      >
        {showNew ? role2 : role}
      </span>
    </div>
  );
};

const Arrow: React.FC<{dir: number; turn: number}> = ({dir, turn}) => {
  const frame = useCurrentFrame();
  const dots = Array.from({length: 7});
  return (
    <div style={{position: 'absolute', left: 800, top: 568, width: 320, height: 100}}>
      <svg width={320} height={100} style={{transform: `rotateY(${turn * 180}deg)`, overflow: 'visible'}}>
        {dots.map((_, i) => {
          const phase = ((frame * 2.2 * dir + i * 44) % 308 + 308) % 308;
          return <circle key={i} cx={phase} cy={50} r={7} fill={C.orange} opacity={0.35 + 0.65 * Math.sin((phase / 308) * Math.PI)} />;
        })}
        <path d="M 280 18 L 318 50 L 280 82" fill="none" stroke={C.orange} strokeWidth={10} strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div
        style={{
          position: 'absolute',
          top: -40,
          width: 320,
          textAlign: 'center',
          fontFamily: FONT.mono,
          fontWeight: 700,
          fontSize: 20,
          letterSpacing: '0.14em',
          color: C.orange,
        }}
      >
        TEACHES
      </div>
    </div>
  );
};

export const Flip: React.FC = () => {
  const frame = useCurrentFrame();
  const turn = ramp(frame, FLIP_AT, FLIP_AT + 22);
  const flipped = frame >= FLIP_AT + 11;

  // Before the flip the human leads and the robot copies 14 frames later.
  // After it the robot leads and the human follows, limbs red until they match.
  const humanKeys: [number, string][] = [
    [0, 'ready'],
    [28, 'guard'],
    [76, 'horse'],
    [124, 'crane'],
    [232, 'horse'],
    [292, 'archer'],
  ];
  const robotKeys: [number, string][] = [
    [0, 'ready'],
    [42, 'guard'],
    [90, 'horse'],
    [138, 'crane'],
    [206, 'horse'],
    [270, 'archer'],
  ];
  const human = poseAt(frame, humanKeys, 18);
  let target = 'ready';
  robotKeys.forEach(([f, n]) => {
    if (frame >= f) target = n;
  });
  const goal = POSES[target];
  const limbColor: Record<string, string> = {};
  if (flipped) {
    LIMBS.forEach((l) => {
      const d = diffDeg(angle(human, l.from as keyof Pose, l.to as keyof Pose), angle(goal, l.from as keyof Pose, l.to as keyof Pose));
      limbColor[l.id] = d < 12 ? C.green : C.red;
    });
  }
  const allGreen = flipped && Object.values(limbColor).every((c) => c === C.green);

  const oldOut = FLIP_AT - 6;
  return (
    <AbsoluteFill>
      <Backdrop />
      {/* Old-way headline */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 70, textAlign: 'center'}}>
        <FadeUp delay={2} out={oldOut}>
          <Label>The old way</Label>
          <Headline size={80} style={{marginTop: 14}}>
            For years, robots have learned by copying us.
          </Headline>
          <Body size={30} style={{marginTop: 16}}>
            Motion capture, remote control, imitation learning: the human is always the teacher.
          </Body>
        </FadeUp>
      </div>
      {/* Flipped headline */}
      <div style={{position: 'absolute', left: 0, right: 0, top: 70, textAlign: 'center'}}>
        <FadeUp delay={FLIP_AT + 8}>
          <Label>We flipped it</Label>
          <Headline size={80} style={{marginTop: 14}}>
            Now the robot <span style={{color: C.orange}}>teaches</span>, and you learn.
          </Headline>
          <Body size={30} style={{marginTop: 16}}>
            It shows a move, watches you copy it, and marks every limb: <span style={{color: C.green}}>green</span> is right,{' '}
            <span style={{color: C.red}}>red</span> needs fixing.
          </Body>
        </FadeUp>
      </div>

      <FadeUp delay={10} dy={50}>
        <Panel x={220} tag={flipped ? 'STUDENT · YOU' : 'TEACHER · HUMAN'} tagColor={flipped ? C.muted : C.orange} border={allGreen ? C.green : C.line}>
          <svg width={PANEL_W} height={PANEL_H}>
            <Figure pose={human} variant="human" x={PANEL_W / 2} y={292} scale={0.9} limbColor={limbColor} />
          </svg>
        </Panel>
      </FadeUp>
      <FadeUp delay={16} dy={50}>
        <Panel x={1140} tag={flipped ? 'TEACHER · UNITREE G1' : 'STUDENT · UNITREE G1'} tagColor={flipped ? C.orange : C.muted}>
          <RobotPanel keys={robotKeys} />
        </Panel>
      </FadeUp>
      <FadeUp delay={22}>
        <Arrow dir={1} turn={turn} />
      </FadeUp>
      <FadeUp delay={26}>
        <RoleLabel x={220} who="Human" role="teacher" role2="student" flip={turn} />
        <RoleLabel x={1140} who="Robot" role="student" role2="teacher" flip={turn} />
      </FadeUp>
    </AbsoluteFill>
  );
};
