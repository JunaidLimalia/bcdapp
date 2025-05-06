import React from "react";

export const Input = React.forwardRef((props, ref) => {
  return (
    <input
      ref={ref}
      {...props}
      className={`border p-2 rounded w-full ${props.className || ""}`}
    />
  );
});

Input.displayName = "Input";