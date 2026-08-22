import { useMemo, useState } from 'react';
import type { JSX } from 'react';
import { useAppStore } from '../state/store';
import { LESSONS } from '../lessons/index';
import { mergePreset } from '../lessons/types';
import type { Lesson } from '../lessons/types';

/*
 * The lessons (§7).
 *
 * "Never gate anything behind a wrong answer; let people poke."
 *
 * So the explanation is revealed by a button as well as by succeeding. The
 * success predicate marks the lesson done and offers the explanation, but a
 * reader who wants to read first is not stopped, and a reader who wanders off
 * the preset entirely is not scolded.
 */

export function LessonsPanel(): JSX.Element {
  const activeId = useAppStore((s) => s.activeLessonId);
  const completed = useAppStore((s) => s.completedLessons);
  const applyLesson = useAppStore((s) => s.applyLesson);
  const clearLesson = useAppStore((s) => s.clearLesson);
  const [openId, setOpenId] = useState<string | null>(null);

  const active = LESSONS.find((l) => l.id === activeId) ?? null;

  return (
    <div className="px-4 py-3">
      <div className="mb-3 flex items-baseline justify-between">
        <p className="panel-title">Lessons</p>
        <span className="num text-[11px] text-[var(--color-text-lo)]">
          {completed.length}/{LESSONS.length}
        </span>
      </div>

      {active !== null && <ActiveLesson lesson={active} onExit={clearLesson} />}

      <div className="flex flex-col gap-px">
        {LESSONS.map((lesson) => {
          const isActive = lesson.id === activeId;
          const isDone = completed.includes(lesson.id);
          const isOpen = openId === lesson.id;
          return (
            <div key={lesson.id}>
              <button
                className="flex w-full items-baseline gap-2 py-1.5 text-left"
                onClick={() => setOpenId(isOpen ? null : lesson.id)}
                aria-expanded={isOpen}
              >
                <span className="num w-5 shrink-0 text-[11px] text-[var(--color-text-lo)]">
                  {lesson.number.toString().padStart(2, '0')}
                </span>
                <span
                  className="flex-1 text-[12px]"
                  style={{
                    color: isActive
                      ? 'var(--color-text-hi)'
                      : isDone
                        ? 'var(--color-text-mid)'
                        : 'var(--color-text-mid)',
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  {lesson.title}
                </span>
                {isDone && (
                  <span
                    className="num shrink-0 text-[11px]"
                    style={{ color: 'var(--color-weight-positive)' }}
                    title="Success condition met"
                  >
                    ✓
                  </span>
                )}
              </button>

              {isOpen && (
                <div className="pb-2 pl-7">
                  <p className="mb-2 text-[11px] leading-relaxed text-[var(--color-text-mid)]">
                    {lesson.goal}
                  </p>
                  <button
                    className="control w-full"
                    onClick={() => applyLesson(lesson.id, lesson.preset)}
                  >
                    {isActive ? 'Reload this lesson' : 'Load this lesson'}
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ActiveLesson({ lesson, onExit }: { lesson: Lesson; onExit: () => void }): JSX.Element {
  const metrics = useAppStore((s) => s.metrics);
  const latest = useAppStore((s) => s.latest);
  const network = useAppStore((s) => s.network);
  const epoch = useAppStore((s) => s.epoch);
  const status = useAppStore((s) => s.trainingStatus);
  const applyLesson = useAppStore((s) => s.applyLesson);
  const markComplete = useAppStore((s) => s.markLessonComplete);
  const completed = useAppStore((s) => s.completedLessons);
  const [revealed, setRevealed] = useState(false);

  const succeeded = useMemo(() => {
    const ok = lesson.successPredicate({ metrics, latest, network, epoch, status });
    if (ok) markComplete(lesson.id);
    return ok;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lesson, metrics, latest, network, epoch, status]);

  const showExplanation = succeeded || revealed || completed.includes(lesson.id);

  return (
    <div className="mb-4 border-l-2 border-[var(--color-weight-positive)] pl-3">
      <p className="mb-1 text-[12px] font-semibold text-[var(--color-text-hi)]">{lesson.title}</p>
      <p className="mb-2 text-[11px] leading-relaxed text-[var(--color-text-mid)]">{lesson.goal}</p>

      <p className="label mb-1">what to watch</p>
      <ul className="mb-3 list-disc pl-4">
        {lesson.whatToWatch.map((line) => (
          <li key={line} className="text-[11px] leading-relaxed text-[var(--color-text-mid)]">
            {line}
          </li>
        ))}
      </ul>

      {lesson.variants !== undefined && lesson.variants.length > 0 && (
        <>
          <p className="label mb-1">try also</p>
          <div className="mb-3 flex flex-col gap-1.5">
            {lesson.variants.map((variant) => (
              <div key={variant.label}>
                <button
                  className="control w-full"
                  onClick={() => applyLesson(lesson.id, mergePreset(lesson.preset, variant.preset))}
                >
                  {variant.label}
                </button>
                <p className="mt-0.5 text-[10px] leading-snug text-[var(--color-text-lo)]">
                  {variant.note}
                </p>
              </div>
            ))}
          </div>
        </>
      )}

      <div
        className="mb-2 flex items-baseline justify-between gap-2 border-t border-[var(--color-line-hair)] pt-2"
        style={{ color: succeeded ? 'var(--color-weight-positive)' : 'var(--color-text-lo)' }}
      >
        <span className="text-[11px]">{lesson.successLabel}</span>
        <span className="num text-[11px]">{succeeded ? '✓' : '—'}</span>
      </div>

      {showExplanation ? (
        <p className="text-[11px] leading-relaxed text-[var(--color-text-mid)]">
          {lesson.explanation}
        </p>
      ) : (
        // Never gated behind a wrong answer: the reader can simply ask.
        <button
          className="control w-full"
          onClick={() => setRevealed(true)}
          title="You do not have to succeed first"
        >
          Read the explanation
        </button>
      )}

      <button className="control mt-2 w-full" onClick={onExit}>
        Leave lesson
      </button>
    </div>
  );
}
