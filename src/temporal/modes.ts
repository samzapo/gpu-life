type ModeSwitch = 'game' | 'tour';

export function modeEnabled(key: ModeSwitch) {
  const value = new URL(location.href).searchParams.get(key);
  if (key === 'game') return value === '1' || value === 'true';
  return value !== '0' && value !== 'false';
}

export function saveMode(key: ModeSwitch, enabled: boolean) {
  const url = new URL(location.href);
  if (key === 'game') {
    if (enabled) url.searchParams.set(key, '1');
    else url.searchParams.delete(key);
  } else {
    if (enabled) url.searchParams.delete(key);
    else url.searchParams.set(key, '0');
  }
  if (url.href !== location.href) history.replaceState(null, '', url);
}
