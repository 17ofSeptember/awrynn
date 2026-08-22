/*
 * Regenerate docs/LESSONS.md from the lesson data.
 *
 * Generated rather than written by hand, because a lesson's pedagogy and its
 * success criteria are already stated in src/lessons/index.ts and a second
 * prose copy would drift from the code that actually runs.
 *
 *   npm run docs:lessons
 */

import { writeFileSync } from 'node:fs';
import { LESSONS } from '../src/lessons/index';

const lines: string[] = [];

lines.push('# LESSONS.md');
lines.push('');
lines.push('Each lesson\'s pedagogy and success criteria.');
lines.push('');
lines.push(
  '> **Generated from `src/lessons/index.ts` by `npm run docs:lessons`.** Lessons are data, not prose, and a hand-written copy of them would drift from the code that runs. Edit the lessons and regenerate.',
);
lines.push('');
lines.push(
  'Every preset below stores an explicit seed. `src/lessons/__tests__/lessons.test.ts` trains each configuration headlessly and asserts the lesson still demonstrates what it claims, which is the Phase 7 gate in §11.',
);
lines.push('');

lines.push('| # | Lesson | Success condition |');
lines.push('| --- | --- | --- |');
for (const lesson of LESSONS) {
  lines.push(`| ${lesson.number} | [${lesson.title}](#${lesson.id}) | ${lesson.successLabel} |`);
}
lines.push('');

for (const lesson of LESSONS) {
  const a = lesson.preset.architecture;
  const shape = [a.inputSize, ...a.layers.map((l) => l.units)].join('-');
  const activations = [...new Set(a.layers.map((l) => l.activation))].join(', ');

  lines.push('---');
  lines.push('');
  lines.push(`## ${lesson.number}. ${lesson.title}`);
  lines.push('');
  lines.push(`<a id="${lesson.id}"></a>`);
  lines.push('');
  lines.push(lesson.goal);
  lines.push('');

  lines.push('**Preset**');
  lines.push('');
  lines.push('| | |');
  lines.push('| --- | --- |');
  lines.push(`| architecture | \`${shape}\` (${activations}) |`);
  lines.push(`| loss | \`${a.loss}\` |`);
  lines.push(`| init | \`${a.init.kind}\` |`);
  lines.push(`| network seed | \`${a.seed}\` |`);
  lines.push(
    `| dataset | \`${lesson.preset.dataset.name}\`, ${lesson.preset.dataset.samples ?? '?'} samples, noise ${lesson.preset.dataset.noise ?? 0}, seed \`${lesson.preset.dataset.seed ?? 0}\` |`,
  );
  lines.push(
    `| training | ${lesson.preset.training.optimizer.name}, lr ${lesson.preset.training.learningRate}, batch ${lesson.preset.training.batchSize}, ${lesson.preset.training.maxEpochs} epochs |`,
  );
  if (a.l2 > 0) lines.push(`| L2 | ${a.l2} |`);
  if (lesson.preset.training.dropout > 0) lines.push(`| dropout | ${lesson.preset.training.dropout} |`);
  if (lesson.preset.training.standardize) lines.push('| standardize | on |');
  if (lesson.preset.dataset.featureScale !== undefined) {
    lines.push(`| feature scale | \`[${lesson.preset.dataset.featureScale.join(', ')}]\` |`);
  }
  lines.push('');

  lines.push('**What to watch**');
  lines.push('');
  for (const item of lesson.whatToWatch) lines.push(`- ${item}`);
  lines.push('');

  if (lesson.variants !== undefined && lesson.variants.length > 0) {
    lines.push('**Variants**');
    lines.push('');
    for (const variant of lesson.variants) lines.push(`- **${variant.label}.** ${variant.note}`);
    lines.push('');
  }

  lines.push(`**Success:** ${lesson.successLabel}`);
  lines.push('');
  lines.push('**Explanation**');
  lines.push('');
  lines.push(lesson.explanation);
  lines.push('');

  lines.push('**Verified by**');
  lines.push('');
  for (const evidence of lesson.evidence) lines.push(`- ${evidence.label}`);
  lines.push('');
}

writeFileSync('docs/LESSONS.md', `${lines.join('\n')}\n`);
process.stdout.write(`docs/LESSONS.md written: ${LESSONS.length} lessons\n`);
