/**
 * Global motion setup. Loaded once from layout/theme.liquid, after
 * gsap.min.js, ScrollTrigger.min.js, SplitText.min.js and lenis.min.js.
 * Sections and blocks must never load GSAP themselves.
 *
 * Built-in reveals, enabled with an attribute in any section or block:
 *   data-motion="lines"  Headline reveals line by line (SplitText, masked).
 *   data-motion="clip"   Image reveals with a clip path wipe.
 *   data-motion="rise"   Children fade and rise in a stagger as they scroll into view.
 *   data-motion="up"     The element itself fades and rises into view.
 *   data-parallax        Added to a "clip" element, its image also drifts with scroll.
 *
 * Custom animations from a section's {% javascript %} tag:
 *   const onMotion = (fn) =>
 *     window.Motion ? fn(window.Motion) : document.addEventListener('motion:ready', () => fn(window.Motion), { once: true });
 *   onMotion((Motion) => Motion.add(el, (el) => { gsap.from(el, { opacity: 0 }); }));
 *
 * Callbacks passed to Motion.add only run when prefers-reduced-motion is not
 * set. Everything they create is reverted when reduced motion is switched on
 * or when the section is reloaded in the theme editor. An optional third
 * argument adds a media query, e.g. Motion.add(el, setup, '(min-width: 750px)').
 */
(() => {
  if (window.Motion) return;

  const root = document.documentElement;
  const { gsap, ScrollTrigger, SplitText, Lenis } = window;

  if (!gsap || !ScrollTrigger || !SplitText) {
    root.classList.remove('motion-ok');
    return;
  }

  const MOTION_OK = '(prefers-reduced-motion: no-preference)';
  const TRIGGER_START = 'top 85%';

  gsap.registerPlugin(ScrollTrigger, SplitText);
  gsap.defaults({ ease: 'power3.out', duration: 1.1 });

  /* Smooth scroll */

  let lenis = null;
  const raf = (time) => lenis.raf(time * 1000);

  function startLenis() {
    // The theme editor scrolls the preview to the selected section, which Lenis would fight.
    if (!Lenis || window.Shopify?.designMode) return;

    lenis = new Lenis({ anchors: true });
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add(raf);
    gsap.ticker.lagSmoothing(0);
  }

  function stopLenis() {
    if (!lenis) return;

    gsap.ticker.remove(raf);
    gsap.ticker.lagSmoothing(500, 33);
    lenis.destroy();
    lenis = null;
  }

  gsap.matchMedia().add(MOTION_OK, () => {
    root.classList.add('motion-ok');
    startLenis();

    return () => {
      root.classList.remove('motion-ok');
      stopLenis();
    };
  });

  /*
   * Cursor follower and magnetic elements, for mouse and trackpad only.
   * The native cursor stays visible; the follower is decoration.
   *   [data-cursor="Label"]  The follower grows and shows the label.
   *   [data-magnetic]        The element drifts toward the pointer.
   */

  gsap.matchMedia().add(`${MOTION_OK} and (hover: hover) and (pointer: fine)`, () => {
    const cursor = document.createElement('div');
    cursor.className = 'cursor';
    cursor.setAttribute('aria-hidden', 'true');
    cursor.innerHTML = '<span class="cursor__label"></span>';
    document.body.append(cursor);

    const label = cursor.firstElementChild;
    const moveX = gsap.quickTo(cursor, 'x', { duration: 0.45, ease: 'power3' });
    const moveY = gsap.quickTo(cursor, 'y', { duration: 0.45, ease: 'power3' });
    gsap.set(cursor, { xPercent: -50, yPercent: -50, scale: 0.12, autoAlpha: 0 });

    let visible = false;
    let state = '';
    let magnet = null;

    const setState = (target) => {
      const labelled = target.closest('[data-cursor]');
      const interactive = target.closest('a, button, summary, label, input, select');
      const next = labelled ? `label:${labelled.dataset.cursor}` : interactive ? 'hover' : 'idle';
      if (next === state) return;

      state = next;
      label.textContent = labelled ? labelled.dataset.cursor : '';
      cursor.classList.toggle('cursor--label', Boolean(labelled));
      // Over links and buttons a solid dot would hide their text: show a ring instead.
      cursor.classList.toggle('cursor--ring', Boolean(interactive) && !labelled);
      gsap.to(cursor, { scale: labelled ? 1 : interactive ? 0.4 : 0.12, duration: 0.5, overwrite: 'auto' });
    };

    const release = () => {
      if (!magnet) return;
      gsap.to(magnet, { x: 0, y: 0, duration: 1, ease: 'elastic.out(1, 0.4)', overwrite: 'auto' });
      magnet = null;
    };

    const attract = (event) => {
      const el = event.target.closest('[data-magnetic]');
      if (magnet && magnet !== el) release();
      if (!el) return;

      magnet = el;
      const rect = el.getBoundingClientRect();
      const dx = event.clientX - (rect.left + rect.width / 2);
      const dy = event.clientY - (rect.top + rect.height / 2);
      gsap.to(el, { x: dx * 0.25, y: dy * 0.35, duration: 0.6, overwrite: 'auto' });
    };

    const onMove = (event) => {
      if (!visible) {
        visible = true;
        gsap.set(cursor, { x: event.clientX, y: event.clientY });
        gsap.to(cursor, { autoAlpha: 1, duration: 0.3 });
      }

      moveX(event.clientX);
      moveY(event.clientY);
      setState(event.target);
      attract(event);
    };

    const onLeave = () => {
      visible = false;
      gsap.to(cursor, { autoAlpha: 0, duration: 0.3 });
      release();
    };

    document.addEventListener('pointermove', onMove, { passive: true });
    document.documentElement.addEventListener('pointerleave', onLeave);

    return () => {
      document.removeEventListener('pointermove', onMove);
      document.documentElement.removeEventListener('pointerleave', onLeave);
      release();
      cursor.remove();
    };
  });

  /*
   * Page transition. A curtain rises to cover the page before following an
   * internal link. The next page uncovers itself with a CSS animation
   * (html.page-arriving, set in theme.liquid), so its content never waits on JS.
   */

  const curtain = document.querySelector('.page-curtain');
  const TRANSITION_KEY = 'page-transition';

  function shouldTransition(link, event) {
    if (!link || event.defaultPrevented || event.button !== 0) return false;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return false;
    if (link.target && link.target !== '_self') return false;
    if (link.hasAttribute('download') || link.matches('[data-no-transition], [data-cart-drawer-trigger]')) return false;
    // Open modals sit in the top layer, above the curtain.
    if (link.closest('dialog')) return false;

    const url = new URL(link.href, window.location.href);
    if (url.origin !== window.location.origin) return false;

    // Same page, possibly with a hash: let the browser (or Lenis) handle it.
    return url.pathname !== window.location.pathname || url.search !== window.location.search;
  }

  if (curtain && !window.Shopify?.designMode) {
    curtain.addEventListener('animationend', () => root.classList.remove('page-arriving'));

    gsap.matchMedia().add(MOTION_OK, () => {
      gsap.set(curtain, { y: 0, yPercent: 100 });

      const onClick = (event) => {
        const link = event.target.closest('a[href]');
        if (!shouldTransition(link, event)) return;

        event.preventDefault();
        root.classList.remove('page-arriving');

        gsap.to(curtain, {
          yPercent: 0,
          duration: 0.7,
          ease: 'power3.inOut',
          onComplete: () => {
            try {
              sessionStorage.setItem(TRANSITION_KEY, '1');
            } catch {}
            window.location.assign(link.href);
          },
        });
      };

      // Back/forward cache restores the page as it was left: covered.
      const onPageShow = (event) => {
        if (event.persisted) gsap.set(curtain, { yPercent: 100 });
      };

      document.addEventListener('click', onClick);
      window.addEventListener('pageshow', onPageShow);

      return () => {
        document.removeEventListener('click', onClick);
        window.removeEventListener('pageshow', onPageShow);
      };
    });
  }

  /* Scoped animations */

  // Each scope gets its own matchMedia so it can be reverted on its own.
  const scopes = new Map();

  // A one-time reveal whose start line sits past the page's maximum scroll
  // (content near the end of a short page) could never fire and would stay
  // hidden. After each refresh, play those immediately.
  ScrollTrigger.addEventListener('refresh', () => {
    const maxScroll = ScrollTrigger.maxScroll(window);

    ScrollTrigger.getAll().forEach((trigger) => {
      if (!trigger.vars.once || trigger.start <= maxScroll) return;

      if (trigger.animation) trigger.animation.play();
      else trigger.vars.onEnter?.(trigger);
    });
  });

  // Sections register in script order, not page order. Sorting before a
  // refresh makes triggers below a pinned section account for its spacer.
  let refreshQueued = false;

  function queueRefresh() {
    if (refreshQueued) return;

    refreshQueued = true;
    requestAnimationFrame(() => {
      refreshQueued = false;
      ScrollTrigger.sort();
      ScrollTrigger.refresh();
    });
  }

  // Content that changes height after load (quiz results, opened FAQ answers,
  // late-loading recommendations) shifts everything below it. Without a
  // re-measure, a pinned section further down keeps its old positions and
  // pins on top of that content. So re-measure whenever the page height
  // settles at a new value. Pin spacers are part of that height but stay the
  // same size on a refresh, so this doesn't loop.
  const main = document.querySelector('main') || document.body;
  let measuredHeight = main.offsetHeight;
  let heightTimer;

  new ResizeObserver(() => {
    clearTimeout(heightTimer);
    heightTimer = setTimeout(() => {
      const height = main.offsetHeight;
      if (Math.abs(height - measuredHeight) < 2) return;

      measuredHeight = height;
      queueRefresh();
      lenis?.resize();
    }, 150);
  }).observe(main);

  ScrollTrigger.addEventListener('refresh', () => {
    measuredHeight = main.offsetHeight;
  });

  function add(scope, setup, query = '') {
    const mm = gsap.matchMedia();
    mm.add(query ? `${MOTION_OK} and ${query}` : MOTION_OK, () => setup(scope), scope);

    if (!scopes.has(scope)) scopes.set(scope, []);
    scopes.get(scope).push(mm);
    queueRefresh();

    return mm;
  }

  function revert(container) {
    for (const [scope, list] of scopes) {
      if (!container.contains(scope)) continue;

      list.forEach((mm) => mm.revert());
      scopes.delete(scope);
    }
  }

  /* Built-in reveals */

  const reveals = {
    lines(el) {
      SplitText.create(el, {
        type: 'lines',
        mask: 'lines',
        autoSplit: true,
        onSplit(self) {
          gsap.set(el, { visibility: 'visible' });

          return gsap.from(self.lines, {
            yPercent: 100,
            stagger: 0.08,
            scrollTrigger: { trigger: el, start: TRIGGER_START, once: true },
          });
        },
      });
    },

    clip(el) {
      const media = el.querySelector('img, video, svg');
      const parallax = el.hasAttribute('data-parallax');
      const scrollTrigger = { trigger: el, start: TRIGGER_START, once: true };

      gsap.set(el, { visibility: 'visible' });
      gsap.fromTo(
        el,
        { clipPath: 'inset(100% 0% 0% 0%)' },
        { clipPath: 'inset(0% 0% 0% 0%)', duration: 1.3, ease: 'power4.inOut', scrollTrigger }
      );

      if (!media) return;

      // With parallax the media stays slightly zoomed so its drift never shows an edge.
      gsap.fromTo(media, { scale: parallax ? 1.3 : 1.15 }, { scale: parallax ? 1.12 : 1, duration: 1.8, scrollTrigger });

      if (parallax) {
        gsap.fromTo(
          media,
          { yPercent: -5 },
          {
            yPercent: 5,
            ease: 'none',
            scrollTrigger: { trigger: el, start: 'top bottom', end: 'bottom top', scrub: true },
          }
        );
      }
    },

    up(el) {
      gsap.fromTo(
        el,
        { autoAlpha: 0, y: 32 },
        { autoAlpha: 1, y: 0, scrollTrigger: { trigger: el, start: TRIGGER_START, once: true } }
      );
    },

    rise(el) {
      const items = [...el.children];

      gsap.set(el, { visibility: 'visible' });
      gsap.set(items, { autoAlpha: 0, y: 48 });

      // Items entering together (one row) stagger as a group.
      ScrollTrigger.batch(items, {
        start: 'top 90%',
        once: true,
        onEnter: (batch) => gsap.to(batch, { autoAlpha: 1, y: 0, stagger: 0.12 }),
      });
    },
  };

  function init(container = document) {
    container.querySelectorAll('[data-motion]').forEach((el) => {
      const reveal = reveals[el.dataset.motion];
      if (reveal && !scopes.has(el)) add(el, reveal);
    });
  }

  /* Theme editor */

  document.addEventListener('shopify:section:load', (event) => {
    init(event.target);
    ScrollTrigger.refresh();
  });

  document.addEventListener('shopify:section:unload', (event) => revert(event.target));

  window.Motion = {
    add,
    revert,
    init,
    get lenis() {
      return lenis;
    },
  };

  init();
  document.dispatchEvent(new CustomEvent('motion:ready'));
})();
