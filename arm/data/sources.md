# Sign sources (hx-data, 17 Oct 2026)

How the handshapes were verified: BSL SignBank (https://bslsignbank.ucl.ac.uk/) lists the number signs
and regional variants but its sign pages are video only. They carry no text handshape, and its
handshape search needs a login or JavaScript, so the finger-by-finger descriptions come from the
dictionary at british-sign.co.uk (Lisia Digital Limited, a commercial BSL course site). It is NOT
SignBank, the BDA or Signature, and it is not a fluent-signer review. Each entry says "illustration is a
starting point, watch video". Spread and thumb values in `signs.json` are my estimates where the source
is silent (noted per sign).

## Used (in `signs.json`)
- one: https://www.british-sign.co.uk/bsl-dictionary/one ("Index finger extended"); SignBank https://bslsignbank.ucl.ac.uk/dictionary/regional/one-1.html
- two: https://www.british-sign.co.uk/bsl-dictionary/two ("Index and middle fingers extended"); SignBank .../regional/two-1.html
- three: https://www.british-sign.co.uk/bsl-dictionary/three ("Index, middle, and ring fingers extended"); SignBank .../regional/three-1.html
- five: https://www.british-sign.co.uk/bsl-dictionary/five ("All fingers and thumb extended"); SignBank .../regional/five-1.html
- seven: https://www.british-sign.co.uk/bsl-dictionary/seven ("Thumb and index finger extended - held horizontal"); SignBank .../regional/seven-1.html
- sorry: https://www.british-sign.co.uk/bsl-dictionary/sorry ("Closed hand makes a circular motion on chest"); not checked on SignBank

## Used in statements (`statements.json`, not in the sign set)
- me: https://www.british-sign.co.uk/bsl-dictionary/me ("Index finger points at signer", also covers I). Same
  handshape as one, so it stays out of the sign set; inside "I'm sorry." (ME SORRY) it is taught as
  `approximated`, handshape only: the hand has no wrist to point at itself and the direction is not checked.
- The statement order ME SORRY is our gloss, not from a source and not reviewed by a fluent signer.
- Phrasebook signs for typed statements (all `approximated`, handshape only; descriptions quoted from
  https://www.british-sign.co.uk/bsl-dictionary/<id>, words mapped only where that page lists them):
  - you ("Point with index finger at the person being referred to."): pointing handshape, direction not checked.
  - my ("Closed hand held to chest.", also covers mine): fist, chest location not checked.
  - good ("Make short forward movement using closed hand with thumb up (1 or 2 hands)", also covers great).
  - thank-you ("Flat hand starts with fingertips on chin. Hand moves down and away from signer.", also covers thanks, please).
  - hello ("Wave hand near side of head.", also covers hi, goodbye).
  - want ("Flat dominant hand moves down chest then twists out.", also covers need).
  - morning ("Dominant hand bent. Fingertips tap far side of chest then near side of chest."): good morning = GOOD MORNING.
  - afternoon, name, deaf ('N' shape: index and middle fingers, at chin / forehead / ear): same handshape as two.
  - drink ("Dominant hand mimes drinking out of a glass.", also covers glass, tumbler, beer): C-shape is my estimate.
  - Two-handed on that site, so skipped with a reason (`cannot_sign`): night, welcome.
  - Not found on that site (404), so not in the phrasebook: yes, no, fine, like, evening, today.
- Typed statements are signed in the typed English word order (closer to Sign Supported English than BSL
  grammar); words with no sign are listed on screen and skipped, never guessed.

## Kept in `signs.json` as `not_representable`
- four ("Index, middle, ring, and little fingers extended"): needs the little finger; identical to three on this hand.
- nine ("Thumb, index, middle, and ring fingers extended - held horizontal"): identical to five here apart from orientation.

## Verified but not used (spare candidates, not in the JSON)
- six: "Closed hand with bent thumb held up." Thumb only differs from the fist in sorry; unreliable if the thumb is dropped.
- eight: "Thumb, index, and middle fingers extended - held horizontal." Thumb-dependent, orientation not checked.
- good: "Short forward movement using closed hand with thumb up." Same handshape as six, movement not checked.

## Rejected and why
- Fingerspelling letters (two-handed in BSL).
- father ("Fingerspelled F"), mother ("Fingerspelled M taps palm"), car, love: two hands.
- eat (bunched fingers), tea, bird, money, promise, snake, sleep: need the little finger, a pinch or flick, or both hands; location or movement is the sign.
- hello, nice: handshape is not the defining part (wave near head; thumb brushes chin).
- you: same handshape as one; the meaning is the pointing direction, which is not checked.
- ten: no handshape text found in the source.
