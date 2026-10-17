import {loadFont as loadDisplay} from '@remotion/google-fonts/BricolageGrotesque';
import {loadFont as loadBody} from '@remotion/google-fonts/Inter';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';

const display = loadDisplay('normal', {weights: ['600', '700', '800'], subsets: ['latin']});
const body = loadBody('normal', {weights: ['400', '500', '600', '700'], subsets: ['latin']});
const mono = loadMono('normal', {weights: ['500', '700'], subsets: ['latin']});

export const FONT = {
  display: display.fontFamily,
  body: body.fontFamily,
  mono: mono.fontFamily,
};

// Same palette as the booth app: dark ink, orange "watch me", green = right, red = fix it.
export const C = {
  bg: '#0A0E13',
  panel: '#121922',
  panel2: '#18212C',
  line: '#243140',
  text: '#F3EFE7',
  muted: '#8E99A6',
  dim: '#5C6773',
  orange: '#F5A623',
  green: '#3BD16F',
  red: '#EF4444',
  sky: '#2A4866',
  robot: '#C9CFD6',
  robotDark: '#2B3138',
  skin: '#E8C4A0',
};

export const FPS = 30;
export const W = 1920;
export const H = 1080;
