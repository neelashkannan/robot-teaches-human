import React from 'react';
import {DemoScene, demoFrames} from '../components/Demo';
import {ARM, UNITREE} from '../footage';
import {FPS} from '../theme';

export const ARM_FRAMES = demoFrames(ARM.clips, FPS);
export const UNITREE_FRAMES = demoFrames(UNITREE.clips, FPS);

export const ArmDemo: React.FC = () => (
  <DemoScene label={ARM.label} title={ARM.title} src={ARM.src} aspect={ARM.aspect} clips={ARM.clips} steps={ARM.steps} />
);

export const UnitreeDemo: React.FC = () => (
  <DemoScene
    label="Teacher 2 · virtual Unitree G1 humanoid"
    title="Kung fu stances, limb by limb"
    src={UNITREE.src}
    aspect={UNITREE.aspect}
    clips={UNITREE.clips}
    steps={UNITREE.steps}
  />
);
