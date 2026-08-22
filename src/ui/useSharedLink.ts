import { useEffect, useState } from 'react';
import { useAppStore } from '../state/store';
import { decodeShareLink } from '../state/shareLink';

export interface SharedLinkNotice {
  readonly kind: 'error' | 'info';
  readonly message: string;
}

/**
 * Adopt the state carried in the address bar's fragment.
 *
 * Runs on mount and on `hashchange`, so pasting a link into an already-open tab
 * works: the browser does not reload for a fragment change, and without the
 * listener the paste would appear to do nothing at all.
 *
 * The fragment is left in place afterwards rather than cleared. A URL should
 * describe what you are looking at, which is what makes the link bookmarkable
 * and re-copyable from the address bar. It does mean a reload returns you to
 * the shared network rather than to your edits, but a reload already discards
 * edits, so this takes nothing away.
 *
 * Both outcomes produce a notice, for the same reason.
 *
 * A link that fails must say so: dropping someone into a default network that
 * looks perfectly fine and is not what they were sent is the worst available
 * behaviour.
 *
 * A link that succeeds with weights in it must also say so, because of what it
 * looks like on arrival. The network is trained, the boundary is carved, and
 * the loss chart is empty at epoch 0. That is the truth (the link carries
 * parameters, not five hundred epochs of metrics, and drawing a curve nobody
 * measured would be exactly the kind of invention this project refuses) but
 * without a sentence it reads as a bug.
 */
export function useSharedLink(): { notice: SharedLinkNotice | null; dismiss: () => void } {
  const applySharedState = useAppStore((s) => s.applySharedState);
  const [notice, setNotice] = useState<SharedLinkNotice | null>(null);

  useEffect(() => {
    const consume = (): void => {
      const fragment = window.location.hash;
      // A bare "#" is what an empty anchor leaves behind; it is not a link.
      if (fragment.length <= 1) return;

      const result = decodeShareLink(fragment);
      if (!result.ok) {
        setNotice({ kind: 'error', message: result.error });
        return;
      }

      const error = applySharedState(result.state);
      if (error !== null) {
        setNotice({ kind: 'error', message: error });
        return;
      }
      setNotice({
        kind: 'info',
        message:
          result.state.parameters === null
            ? 'Opened from a shared link. These weights came from the seed, so they are the same ones the sender saw before training.'
            : 'Opened from a shared link, with the sender’s exact weights. The loss curve is not part of a link, so the epoch counter starts at zero; press Train to carry on from here.',
      });
    };

    consume();
    window.addEventListener('hashchange', consume);
    return () => window.removeEventListener('hashchange', consume);
  }, [applySharedState]);

  return { notice, dismiss: () => setNotice(null) };
}
