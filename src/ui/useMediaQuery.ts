import { useEffect, useState } from 'react';

/*
 * Subscribe to a media query.
 *
 * Used to choose between the desktop sidebars and the mobile drawer. The check
 * has to be a real subscription rather than a one-time read: a tablet rotating
 * from portrait to landscape crosses the breakpoint without reloading, and a
 * layout that only measured once would be wrong until the next refresh.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia(query).matches,
  );

  useEffect(() => {
    const list = window.matchMedia(query);
    const update = (): void => setMatches(list.matches);
    update();
    list.addEventListener('change', update);
    return () => list.removeEventListener('change', update);
  }, [query]);

  return matches;
}

/*
 * Two breakpoints, both chosen from content rather than from device names.
 *
 *   < 1024px   Everything lives in the drawer. The build panel needs ~300px and
 *              the canvas stops being readable below ~700px of its own, so
 *              there is no honest way to show a sidebar and still have a
 *              picture worth looking at.
 *
 *   1024-1279  The build panel returns as a sidebar; analysis stays in the
 *              drawer. A tablet in landscape has room for one, not two.
 *
 *   >= 1280    Both sidebars, no drawer. 300 + 320 of chrome still leaves the
 *              canvas over 650px.
 */
export const BUILD_SIDEBAR_QUERY = '(min-width: 1024px)';
export const ANALYSIS_SIDEBAR_QUERY = '(min-width: 1280px)';
