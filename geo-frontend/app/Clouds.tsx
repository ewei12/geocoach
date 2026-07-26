"use client";
import { useEffect, useState, type CSSProperties } from "react";

type Blob = {
  left: number;
  top: number;
  width: number;
  height: number;
  opacity: number;
};

type Cloud = {
  id: number;
  top: number;
  scale: number;
  delay: number;
  duration: number;
  blobs: Blob[];
};

function makeBlobs(baseOpacity: number): Blob[] {
  const count = 3 + Math.floor(Math.random() * 4); // 3–6 blobs per cloud
  const blobs: Blob[] = [];
  let cursor = 0;
  for (let i = 0; i < count; i++) {
    const width = 70 + Math.random() * 100;
    const height = width * (0.55 + Math.random() * 0.3);
    blobs.push({
      left: cursor,
      top: Math.random() * 35,
      width,
      height,
      opacity: baseOpacity * (0.7 + Math.random() * 0.6),
    });
    cursor += width * (0.45 + Math.random() * 0.25); // overlap neighbors
  }
  return blobs;
}

function Clouds({ count = 9 }: { count?: number }) {
  const [clouds, setClouds] = useState<Cloud[]>([]);

  useEffect(() => {
    const generated = Array.from({ length: count }).map((_, i) => {
      const duration = 45 + Math.random() * 40;
      const baseOpacity = 0.06 + Math.random() * 0.1;
      return {
        id: i,
        top: Math.random() * 80,
        scale: 0.6 + Math.random() * 1.1,
        delay: -(Math.random() * duration),
        duration,
        blobs: makeBlobs(baseOpacity),
      };
    });
    setClouds(generated);
  }, [count]);

  return (
    <div
      className="absolute inset-0 overflow-hidden"
      style={{ pointerEvents: "none" }}
    >
      {clouds.map((c) => {
        const width = Math.max(...c.blobs.map((b) => b.left + b.width)) + 30;
        const height = Math.max(...c.blobs.map((b) => b.top + b.height)) + 30;

        return (
          <div
            key={c.id}
            className="absolute cloud-drift"
            style={
              {
                top: `${c.top}%`,
                left: "-25%",
                width: `${width}px`,
                height: `${height}px`,
                animationDelay: `${c.delay}s`,
                animationDuration: `${c.duration}s`,
                "--s": c.scale,
              } as CSSProperties
            }
          >
            {c.blobs.map((b, i) => (
              <div
                key={i}
                className="cloud-blob"
                style={{
                  left: `${b.left}px`,
                  top: `${b.top}px`,
                  width: `${b.width}px`,
                  height: `${b.height}px`,
                  opacity: b.opacity,
                }}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}

export default Clouds;
