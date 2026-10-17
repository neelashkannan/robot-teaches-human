import React from 'react';
import {Composition, Folder} from 'remotion';
import {Pitch, PITCH_FRAMES, SCENES} from './Pitch';
import {FPS, H, W} from './theme';

export const Root: React.FC = () => (
  <>
    <Composition id="Pitch" component={Pitch} durationInFrames={PITCH_FRAMES} fps={FPS} width={W} height={H} />
    <Folder name="Scenes">
      {SCENES.map(({id, frames, C}) => (
        <Composition key={id} id={id} component={C} durationInFrames={frames} fps={FPS} width={W} height={H} />
      ))}
    </Folder>
  </>
);
