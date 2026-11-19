type ModeSwitch = 'game' | 'tour';

export function modeEnabled(key: ModeSwitch) {
  const value = new URL(location.href).searchParams.get(key);
  return value !== '0' && value !== 'false';
}

export function saveMode(key: ModeSwitch, enabled: boolean) {
  const url = new URL(location.href);
  if (enabled) url.searchParams.delete(key);
  else url.searchParams.set(key, '0');
  if (url.href !== location.href) history.replaceState(null, '', url);
}
