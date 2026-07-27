"use client";

import { useEffect, useState } from "react";
import WorldMap from "react-svg-worldmap";

export default function SupportedMap() {
  const [countries, setCountries] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetch(
      `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001"}/supported-countries`,
    )
      .then((res) => res.json())
      .then((data) => {
        setCountries(
          data.countries.map((code: string) => ({
            country: code.toLowerCase(),
            value: 1,
          })),
        );

        setModelInfo(data.model);
      })
      .catch(console.error);
  }, []);

  if (!countries.length) return null;

  return (
    <section className="w-full mt-20 mb-12">
      <div
        className="relative overflow-hidden rounded-3xl p-8"
        style={{
          background:
            "linear-gradient(135deg, rgba(45,55,75,0.95) 0%, rgba(18,25,42,0.98) 100%)",
          border: "1px solid rgba(245,240,230,0.15)",
          boxShadow: "0 8px 40px rgba(0,0,0,0.35)",
        }}
      >
        {/* soft glow */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background:
              "radial-gradient(circle at 50% 45%, rgba(245,230,200,0.12), transparent 60%)",
          }}
        />

        <div className="relative z-10 flex flex-col items-center gap-5">
          <p
            className="text-xs uppercase tracking-[0.25em]"
            style={{
              color: "rgba(245,240,230,0.7)",
              fontFamily: "var(--font-commanding)",
            }}
          >
            Coverage map
          </p>

          <p
            className="text-sm text-center max-w-md"
            style={{
              color: "rgba(245,240,230,0.65)",
            }}
          >
            Verona is trained on road imagery from countries around the world.
          </p>

          <div
            className="w-full max-w-4xl py-4"
            style={{
              filter: "drop-shadow(0 0 18px rgba(245,230,200,0.35))",
              opacity: 0.95,
            }}
          >
            <WorldMap
              color="#F5E6C8"
              backgroundColor="transparent"
              value-suffix=" supported"
              size="responsive"
              data={countries}
            />
          </div>

          {/* {modelInfo && (
            <div
              className="px-5 py-2 rounded-full text-xs uppercase tracking-widest"
              style={{
                color: "#F5F0E6",
                background: "rgba(245,240,230,0.12)",
                border: "1px solid rgba(245,240,230,0.22)",
                backdropFilter: "blur(10px)",
                fontFamily: "var(--font-commanding)",
              }}
            >
              {countries.length} countries ·{" "}
              {modelInfo.examples.toLocaleString()} images
            </div>
          )} */}
        </div>
      </div>
    </section>
  );
}
