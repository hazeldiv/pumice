import { useEffect, useState } from "react";

interface Props {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  integer?: boolean;
  className?: string;
  placeholder?: string;
  disabled?: boolean;
}

export function NumberInput({ value, onChange, min, max, integer, className, placeholder, disabled }: Props) {
  const [text, setText] = useState(String(value));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused && Number(text) !== value) setText(String(value));
  }, [value, focused, text]);

  const handleChange = (raw: string) => {
    const clean = raw.replace(/[^0-9.\-]/g, "");
    setText(clean);
    if (clean === "" || clean === "-" || clean === "." || clean === "-.") return;
    const parsed = Number(clean);
    if (!Number.isFinite(parsed)) return;
    let next = integer ? Math.round(parsed) : parsed;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    onChange(next);
  };

  const handleBlur = () => {
    setFocused(false);
    if (text.trim() === "" || !Number.isFinite(Number(text))) setText(String(value));
  };

  return (
    <input
      type="text"
      inputMode="decimal"
      autoComplete="off"
      spellCheck={false}
      className={className}
      placeholder={placeholder}
      disabled={disabled}
      value={text}
      onChange={(e) => handleChange(e.target.value)}
      onFocus={(e) => {
        setFocused(true);
        e.currentTarget.select();
      }}
      onBlur={handleBlur}
    />
  );
}
