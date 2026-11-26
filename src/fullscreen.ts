export interface FullscreenDocument extends Document {
  webkitFullscreenElement?: Element | null;
  webkitFullscreenEnabled?: boolean;
  webkitExitFullscreen?: () => Promise<void> | void;
}

export interface FullscreenElement extends HTMLElement {
  webkitRequestFullscreen?: () => Promise<void> | void;
}

export function isFullscreenSupported(
  doc: FullscreenDocument = document as FullscreenDocument,
  target: FullscreenElement | null = (doc.documentElement as FullscreenElement | null),
): boolean {
  if (!target) return false;
  const hasRequest =
    typeof target.requestFullscreen === 'function' ||
    typeof target.webkitRequestFullscreen === 'function';
  if (!hasRequest) return false;
  const enabled = doc.fullscreenEnabled ?? doc.webkitFullscreenEnabled;
  return enabled !== false;
}

export function isFullscreenActive(
  doc: FullscreenDocument = document as FullscreenDocument,
): boolean {
  return Boolean(doc.fullscreenElement || doc.webkitFullscreenElement);
}

export async function toggleFullscreen(
  doc: FullscreenDocument = document as FullscreenDocument,
  target: FullscreenElement | null = (doc.documentElement as FullscreenElement | null),
): Promise<boolean> {
  if (isFullscreenActive(doc)) {
    try {
      if (typeof doc.exitFullscreen === 'function') {
        await doc.exitFullscreen();
      } else if (typeof doc.webkitExitFullscreen === 'function') {
        await doc.webkitExitFullscreen();
      }
    } catch {
      // Ignore user/browser fullscreen exit rejections
    }
    return isFullscreenActive(doc);
  }

  if (!target || !isFullscreenSupported(doc, target)) return false;

  try {
    if (typeof target.requestFullscreen === 'function') {
      try {
        await target.requestFullscreen({ navigationUI: 'hide' });
      } catch {
        // Retry without options on browsers rejecting FullscreenOptions
        await target.requestFullscreen();
      }
    } else if (typeof target.webkitRequestFullscreen === 'function') {
      await target.webkitRequestFullscreen();
    }
  } catch {
    // Ignore permission/gesture rejection
  }
  return isFullscreenActive(doc);
}

export interface FullscreenToggleOptions {
  doc?: FullscreenDocument;
  target?: FullscreenElement | null;
  onChange?: (active: boolean) => void;
}

export function wireFullscreenToggle(
  btn: HTMLButtonElement | null,
  opts: FullscreenToggleOptions = {},
): { sync: () => void } {
  const doc = opts.doc ?? (document as FullscreenDocument);
  const target = opts.target ?? (doc.documentElement as FullscreenElement | null);
  const sync = () => {
    if (!btn) return;
    const supported = isFullscreenSupported(doc, target);
    btn.hidden = !supported;
    if (!supported) return;
    const active = isFullscreenActive(doc);
    btn.setAttribute('aria-pressed', String(active));
    btn.dataset.fullscreen = active ? 'on' : 'off';
    btn.title = '';
    btn.removeAttribute?.('title');
    btn.setAttribute('aria-label', 'Toggle immersive view');
    const labelEl = btn.querySelector('.switch-label');
    if (labelEl) {
      labelEl.textContent = '';
    }
  };

  sync();
  if (!btn) return { sync };

  btn.addEventListener('click', () => {
    void toggleFullscreen(doc, target).finally(() => {
      sync();
    });
  });

  const handleChange = () => {
    sync();
    opts.onChange?.(isFullscreenActive(doc));
  };
  doc.addEventListener('fullscreenchange', handleChange);
  doc.addEventListener('webkitfullscreenchange', handleChange);

  return { sync };
}
