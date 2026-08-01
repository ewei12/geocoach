"use client";
import { useDropzone } from "react-dropzone";
import { useState, useEffect, useRef } from "react";
import Clouds from "./Clouds";

import dynamic from "next/dynamic";
const SupportedMap = dynamic(() => import("./SupportedMap"), { ssr: false });

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

type PreviewFile = File & { preview: string };

type VisionPrediction = { label: string; confidence: number };
type VisionData = Record<string, VisionPrediction[]>;

type RoadLine = {
  double_line: boolean;
  pattern: string;
  confidence: number;
};
type RoadRaw = {
  any_markings_detected: boolean;
  white_line?: RoadLine;
  yellow_line?: RoadLine;
};

type PredictedGuess = { country: string; confidence: number };
type Country = { name: string; code: string };

export default function Home() {
  const [file, setFile] = useState<PreviewFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [vision, setVision] = useState<VisionData>({});
  const [roadData, setRoadData] = useState<string[]>([]);
  const [countries, setCountries] = useState<string[]>([]);
  const [loadingStep, setLoadingStep] = useState(0);
  const [roadRaw, setRoadRaw] = useState<RoadRaw | null>(null);
  const [roadNote, setRoadNote] = useState("");
  const [predictedGuesses, setPredictedGuesses] = useState<PredictedGuess[]>(
    [],
  );
  const [reasoning, setReasoning] = useState("");

  // --- feedback state ---
  const [requestId, setRequestId] = useState<string | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState<
    "idle" | "confirmed" | "correcting" | "logged"
  >("idle");
  const [correctionQuery, setCorrectionQuery] = useState("");
  const [countryList, setCountryList] = useState<Country[]>([]);
  const [countrySuggestions, setCountrySuggestions] = useState<Country[]>([]);
  const [loggedCountry, setLoggedCountry] = useState("");

  // --- rate limit ---
  const [rateLimited, setRateLimited] = useState<string | null>(null);

  const ANALYSIS_STEPS = [
    "Scanning for road markings...",
    "Reading any visible signs...",
    "Checking terrain & flora...",
  ];

  const [dockRect, setDockRect] = useState<DOMRect | null>(null);
  const dockRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!file) return;
    const measure = () => {
      if (dockRef.current) {
        setDockRect(dockRef.current.getBoundingClientRect());
      }
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [file]);

  useEffect(() => {
    if (!loading) return;
    setLoadingStep(0);
    const id = setInterval(() => {
      setLoadingStep((s) => (s + 1) % ANALYSIS_STEPS.length);
    }, 1400);
    return () => clearInterval(id);
  }, [loading]);

  // fetch the country list once, for the correction search box
  useEffect(() => {
    fetch(`${API_URL}/countries`)
      .then((res) => res.json())
      .then(setCountryList)
      .catch((err) => console.error("Error fetching countries:", err));
  }, []);

  const onDrop = async (acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (!f) return;

    const limitedUntil = localStorage.getItem("geocoach_rate_limited_until");
    if (limitedUntil && Date.now() < Number(limitedUntil)) {
      setRateLimited("Demo limit reached.");
      return;
    }

    setVision({});
    resetFeedback();

    let settled = false;
    const previewFile = Object.assign(f, {
      preview: URL.createObjectURL(f),
    }) as PreviewFile;

    // only reveal the loading UI if the request is still pending after a short delay
    const revealTimer = setTimeout(() => {
      if (!settled) {
        setFile(previewFile);
        setLoading(true);
      }
    }, 200);

    const formData = new FormData();
    formData.append("file", f);
    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      settled = true;
      clearTimeout(revealTimer);

      if (res.status === 429) {
        const errBody = await res.json().catch(() => ({}) as any);
        const message = errBody.error || "Demo limit reached.";
        setFile(null);
        URL.revokeObjectURL(previewFile.preview);
        setLoading(false);
        setRateLimited(message);
        localStorage.setItem(
          "geocoach_rate_limited_until",
          String(Date.now() + 24 * 60 * 60 * 1000),
        );
        return;
      }

      setFile(previewFile);
      const data = await res.json();
      setReasoning(data.reasoning || "");
      setVision(data.vision_raw || {});
      setRoadData(data.road_markings || []);
      setCountries(data.possible_countries || []);
      setRoadRaw(data.road_markings_raw || null);
      setRoadNote(data.road_markings || "");
      setPredictedGuesses(data.predicted_guesses || []);
      setRequestId(data.request_id || null);
    } catch (error) {
      settled = true;
      clearTimeout(revealTimer);
      console.error("Error uploading file:", error);
      setFile(null);
      URL.revokeObjectURL(previewFile.preview);
    } finally {
      setLoading(false);
    }
  };

  const removeFile = () => {
    if (file?.preview) URL.revokeObjectURL(file.preview);
    setFile(null);
    setVision({});
    setRoadData([]);
    setRoadRaw(null);
    setRoadNote("");
    setPredictedGuesses([]);
    setReasoning("");
    resetFeedback();
  };

  const resetFeedback = () => {
    setRequestId(null);
    setFeedbackStatus("idle");
    setCorrectionQuery("");
    setCountrySuggestions([]);
    setLoggedCountry("");
  };

  const handleFeedbackYes = async () => {
    setFeedbackStatus("confirmed");
    try {
      await fetch(`${API_URL}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ request_id: requestId }),
      });
    } catch (err) {
      console.error("Error confirming prediction:", err);
    }
  };

  const handleFeedbackNo = () => {
    setFeedbackStatus("correcting");
  };

  const handleCorrectionSearch = (value: string) => {
    setCorrectionQuery(value);
    if (!value.trim()) {
      setCountrySuggestions([]);
      return;
    }
    const q = value.toLowerCase();
    setCountrySuggestions(
      countryList.filter((c) => c.name.toLowerCase().includes(q)).slice(0, 50),
    );
  };

  const submitCorrection = async (country: Country) => {
    setCorrectionQuery(country.name);
    setCountrySuggestions([]);
    setLoggedCountry(country.name);
    setFeedbackStatus("logged");
    try {
      await fetch(`${API_URL}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: requestId,
          correct_code: country.code,
        }),
      });
    } catch (err) {
      console.error("Error logging correction:", err);
    }
  };

  const { getRootProps, getInputProps, open, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [] },
    multiple: false,
    noClick: true,
    noKeyboard: true,
  });

  const bigUploadButton = (
    <button
      type="button"
      onClick={open}
      aria-label="Upload a screenshot"
      className="
        w-11 h-11 rounded-full
        flex items-center justify-center shrink-0
        transition-all hover:scale-105 active:scale-95 cursor-pointer
        focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2
      "
      style={{
        background: "rgba(245,240,230,0.14)",
        border: "1px solid rgba(245,240,230,0.28)",
        backdropFilter: "blur(10px)",
        outlineColor: "var(--accent)",
      }}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 20 20"
        fill="none"
        stroke="var(--text-light)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M10 13V3M10 3L5.5 7.5M10 3l4.5 4.5" />
        <path d="M3.5 14.5V16a1 1 0 001 1h11a1 1 0 001-1v-1.5" />
      </svg>
    </button>
  );
  const dragOverlay = isDragActive && (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center pointer-events-none"
      style={{
        background: "rgba(19,27,46,0.55)",
        border: "3px dashed var(--accent)",
      }}
    ></div>
  );

  const rateLimitOverlay = rateLimited && (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{
        background: "rgba(13, 18, 32, 0.85)",
        backdropFilter: "blur(8px)",
      }}
    >
      <div
        className="flex flex-col items-center text-center max-w-md mx-4"
        style={{
          background: "linear-gradient(135deg, rgba(35,43,69,0.9) 0%, rgba(22,28,48,0.9) 100%)",
          border: "1px solid rgba(245,240,230,0.14)",
          borderRadius: "20px",
          padding: "44px 48px",
        }}
      >
        <div
          style={{
            width: 52,
            height: 52,
            borderRadius: "50%",
            background: "rgba(240,140,120,0.14)",
            border: "1px solid rgba(240,140,120,0.3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 18,
          }}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f0947e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9" />
            <polyline points="12 7 12 12 15 14" />
          </svg>
        </div>
        <h2
          style={{
            fontSize: 21,
            fontWeight: 600,
            color: "#f5f0e6",
            margin: "0 0 10px",
            letterSpacing: "-0.01em",
            fontFamily: "var(--font-headline)",
          }}
        >
          Demo limit reached
        </h2>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            fontSize: 12,
            color: "rgba(245,240,230,0.4)",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            borderTop: "1px solid rgba(245,240,230,0.1)",
            paddingTop: 14,
            width: "100%",
          }}
        >
          Resets daily
        </div>
      </div>
    </div>
  );

  // === feedback panel, styled to match the "Evidence" glass-panel ===
  const feedbackPanel = (
    <div
      className="p-6 text-left flex flex-col gap-4 w-full rounded-2xl"
      style={{
        background:
          "linear-gradient(135deg, var(--bg-mid) 0%, var(--bg-deep) 100%)",
        border: "1px solid rgba(245,240,230,0.12)",
        boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      <p
        className="text-xs uppercase tracking-[0.15em] text-center"
        style={{
          color: "rgba(255, 255, 255, 0.85)",
          fontFamily: "var(--font-commanding)",
        }}
      >
        Feedback
      </p>

      {feedbackStatus === "idle" && (
        <>
          <p
            className="text-sm text-center"
            style={{ color: "rgba(255, 255, 255, 0.6)" }}
          >
            Did we get this right?
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={handleFeedbackYes}
              className="flex-1 flex items-center justify-center gap-2 text-sm font-medium py-3 rounded-2xl transition-all duration-200 hover:scale-[1.02] active:scale-[0.97]"
              style={{
                color: "rgb(120, 220, 160)",
                background: "rgba(63,185,111,0.16)",
                border: "1px solid rgba(63,185,111,0.4)",
                backdropFilter: "blur(10px)",
                boxShadow: "0 2px 10px rgba(63,185,111,0.1)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(63,185,111,0.28)";
                e.currentTarget.style.boxShadow =
                  "0 4px 20px rgba(63,185,111,0.25)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(63,185,111,0.16)";
                e.currentTarget.style.boxShadow =
                  "0 2px 10px rgba(63,185,111,0.1)";
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M4 10.5l4 4 8-9" />
              </svg>
              Yes
            </button>
            <button
              type="button"
              onClick={handleFeedbackNo}
              className="flex-1 flex items-center justify-center gap-2 text-sm font-medium py-3 rounded-2xl transition-all duration-200 hover:scale-[1.02] active:scale-[0.97]"
              style={{
                color: "rgb(240, 140, 120)",
                background: "rgba(229,83,61,0.14)",
                border: "1px solid rgba(229,83,61,0.35)",
                backdropFilter: "blur(10px)",
                boxShadow: "0 2px 10px rgba(229,83,61,0.08)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "rgba(229,83,61,0.24)";
                e.currentTarget.style.boxShadow =
                  "0 4px 20px rgba(229,83,61,0.2)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "rgba(229,83,61,0.14)";
                e.currentTarget.style.boxShadow =
                  "0 2px 10px rgba(229,83,61,0.08)";
              }}
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 20 20"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="5" y1="5" x2="15" y2="15" />
                <line x1="15" y1="5" x2="5" y2="15" />
              </svg>
              No
            </button>
          </div>
        </>
      )}

      {feedbackStatus === "confirmed" && (
        <div className="flex justify-center items-center w-full">
          <img
            src="/a49.png"
            width="100"
            height="100"
            className="rounded-md"
            alt="Success"
          />
        </div>
      )}

      {feedbackStatus === "correcting" && (
        <div className="flex flex-col gap-2">
          <p className="text-sm" style={{ color: "rgba(255, 255, 255, 0.6)" }}>
            What's the correct country?
          </p>
          <input
            type="text"
            value={correctionQuery}
            onChange={(e) => handleCorrectionSearch(e.target.value)}
            placeholder="Start typing a country…"
            autoFocus
            className="text-sm px-3 py-2 rounded-lg outline-none"
            style={{
              background: "rgba(245,240,230,0.08)",
              border: "1px solid rgba(245,240,230,0.2)",
              color: "#ffffff",
            }}
          />
          {countrySuggestions.length > 0 && (
            <div
              className="flex flex-col rounded-lg overflow-y-auto"
              style={{
                border: "1px solid rgba(245,240,230,0.15)",
                maxHeight: "200px",
              }}
            >
              {countrySuggestions.map((c) => (
                <button
                  key={c.code}
                  type="button"
                  onClick={() => submitCorrection(c)}
                  className="text-sm text-left px-3 py-2 transition-colors"
                  style={{
                    color: "#ffffff",
                    background: "rgba(245,240,230,0.03)",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "rgba(245,240,230,0.1)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background =
                      "rgba(245,240,230,0.03)")
                  }
                >
                  {c.name}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {feedbackStatus === "logged" && (
        <p className="text-sm" style={{ color: "#ffffff" }}>
          Got it, the correct answer is {loggedCountry}. This helps to improve
          our model!
        </p>
      )}
    </div>
  );

  const LOGO_TEXT = "GEOCOACH";
  const [displayText, setDisplayText] = useState(LOGO_TEXT);
  const [isScrambling, setIsScrambling] = useState(false);

  const handleLogoClick = () => {
    if (isScrambling) return;
    removeFile(); // fires immediately now
    setIsScrambling(true);

    const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let progress = 0;
    const interval = setInterval(() => {
      setDisplayText(
        LOGO_TEXT.split("")
          .map((letter, i) =>
            i < progress
              ? letter
              : chars[Math.floor(Math.random() * chars.length)],
          )
          .join(""),
      );
      progress += 0.4;
      if (progress >= LOGO_TEXT.length) {
        clearInterval(interval);
        setDisplayText(LOGO_TEXT);
        setIsScrambling(false);
      }
    }, 45);
  };

  return (
    <main
      {...getRootProps()}
      className="relative w-full min-h-screen overflow-y-auto outline-none"
      style={{ background: "var(--bg-deep)" }}
    >
      <input {...getInputProps()} />

      <div className="relative z-10 w-full min-h-screen flex justify-center gap-2">
        {/* left: image panel */}
        {/* inside the photo thumbnail wrapper, right after the <img> + gradient */}
        <div className="flex gap-8 items-start justify-center w-full max-w-5xl mx-auto px-6 pt-20">
          {file && (
            <>
              {/* placeholder reserves layout space, invisible */}
              <div
                ref={dockRef}
                className="hidden md:block shrink-0"
                style={{ width: "18rem" }}
              />
              <div
                className="hidden md:flex flex-col items-center gap-6 transition-all duration-700 ease-in-out"
                style={{
                  position: "fixed",
                  zIndex: 25,
                  top: loading || !dockRect ? "50%" : dockRect.top,
                  left: loading || !dockRect ? "50%" : dockRect.left,
                  transform:
                    loading || !dockRect
                      ? "translate(-50%, -50%) scale(1.15)"
                      : "translate(0, 0) scale(1)",
                  width: "18rem",
                  pointerEvents: loading ? "none" : "auto",
                }}
              >
                <div
                  className="relative rounded-lg overflow-hidden"
                  style={{
                    background: "#0d1220",
                    width: "18rem",
                    height: "24rem",
                  }}
                >
                  <img
                    src={file.preview}
                    alt="Uploaded"
                    className="w-full h-full object-cover"
                  />
                  {/* scanning while it's loading */}
                  {loading && (
                    <div className="absolute inset-0 overflow-hidden pointer-events-none">
                      <div
                        className="absolute left-0 w-full h-10"
                        style={{
                          background:
                            "linear-gradient(180deg, transparent 0%, rgba(var(--accent-rgb, 245,240,230),0.35) 50%, transparent 100%)",
                          animation: "scanY 2.2s ease-in-out infinite",
                        }}
                      />
                    </div>
                  )}
                  {/* subtle bottom gradient so text/brackets stay legible */}
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      background:
                        "linear-gradient(180deg, transparent 60%, rgba(0,0,0,0.5) 100%)",
                    }}
                  />

                  {/* corner brackets, viewfinder-style */}
                  <div className="absolute inset-3 pointer-events-none">
                    {[
                      "top-0 left-0 border-t border-l",
                      "top-0 right-0 border-t border-r",
                      "bottom-0 left-0 border-b border-l",
                      "bottom-0 right-0 border-b border-r",
                    ].map((pos, i) => (
                      <div
                        key={i}
                        className={`absolute w-4 h-4 ${pos}`}
                        style={{ borderColor: "var(--accent)" }}
                      />
                    ))}
                  </div>

                  {/* Source tag */}
                  <span
                    className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-[0.15em] px-3 py-1 rounded-full"
                    style={{
                      color: "var(--text-light)",
                      fontFamily: "var(--font-commanding)",
                      background: "rgba(245,240,230,0.14)",
                      border: "1px solid rgba(245,240,230,0.28)",
                      backdropFilter: "blur(10px)",
                    }}
                  >
                    Source
                  </span>

                  {/* remove button — top middle, red, hover */}
                  {!loading && (
                    <button
                      type="button"
                      onClick={removeFile}
                      aria-label="Remove image"
                      className="absolute top-3 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110"
                      style={{
                        background: "rgb(175, 58, 40)",
                        color: "#fff",
                        boxShadow: "0 2px 10px rgba(229,83,61,0.4)",
                      }}
                    >
                      ✕
                    </button>
                  )}
                </div>
                <style jsx>{`
                  @keyframes scanY {
                    0% {
                      top: -10%;
                    }
                    100% {
                      top: 100%;
                    }
                  }
                `}</style>
                {loading && (
                  <div
                    className="flex flex-col items-center gap-3 transition-opacity duration-500"
                    style={{ opacity: loading ? 1 : 0 }}
                  >
                    <p
                      key={loadingStep}
                      className="text-xs tracking-[0.25em] uppercase fade-step"
                      style={{
                        color: "var(--text-muted)",
                        fontFamily: "var(--font-commanding)",
                      }}
                    >
                      {ANALYSIS_STEPS[loadingStep]}
                    </p>
                  </div>
                )}
                {!loading && requestId && (
                  <div
                    className="w-full transition-opacity duration-700"
                    style={{ opacity: !loading && requestId ? 1 : 0 }}
                  >
                    {feedbackPanel}
                  </div>
                )}
              </div>
            </>
          )}

          {!file && (
            <div className="absolute inset-0 z-0 overflow-hidden">
              <div
                className="absolute inset-0"
                style={{
                  background:
                    "linear-gradient(180deg, var(--bg-deep) 0%, var(--bg-mid) 38%, var(--bg-warm) 72%)",
                }}
              />
              <Clouds />
            </div>
          )}

          <header className="fixed top-0 left-0 w-full z-20 px-6 py-4 flex items-start justify-between">
            {!loading && !file && (
              <h1
                className={`text-4xl font-bold logo-text ${isScrambling ? "is-scrambling" : ""}`}
                onClick={handleLogoClick}
                style={{
                  fontFamily: "var(--font-darumadrop)",
                  color: "var(--text-light)",
                  userSelect: "none",
                }}
              >
                {displayText.split("").map((char, i) => (
                  <span
                    key={i}
                    className="logo-letter"
                    style={{
                      ["--i" as any]: i,
                      ...(isScrambling
                        ? { color: "var(--text-light)", transform: "none" }
                        : {}),
                    }}
                  >
                    {char}
                  </span>
                ))}
              </h1>
            )}
            {file && (
              <>
                <span
                  className={`text-4xl font-bold logo-text ${isScrambling ? "is-scrambling" : ""}`}
                  onClick={handleLogoClick}
                  style={{
                    fontFamily: "var(--font-darumadrop)",
                    color: "var(--text-light)",
                    userSelect: "none",
                  }}
                >
                  {displayText.split("").map((char, i) => (
                    <span
                      key={i}
                      className="logo-letter"
                      style={{
                        ["--i" as any]: i,
                        ...(isScrambling
                          ? { color: "var(--text-light)", transform: "none" }
                          : {}),
                      }}
                    >
                      {char}
                    </span>
                  ))}
                </span>
                {bigUploadButton}
              </>
            )}
          </header>
          <div className="relative z-10 flex flex-col w-full max-w-xl">
            {!file ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-10 pb-16">
                <div className="max-w-xl text-center flex flex-col gap-8 px-4 pt-20">
                  {/* ^ title div */}
                  <h2
                    className="text-2xl sm:text-3xl md:text-4xl whitespace-nowrap leading-tight drop-shadow-md"
                    style={{
                      fontFamily: "var(--font-headline)",
                      fontWeight: 600,
                      color: "var(--text-light)",
                    }}
                  >
                    Analyze the roads you explore.
                  </h2>
                </div>

                <button
                  type="button"
                  onClick={open}
                  className="glass-panel flex flex-col items-center gap-3 px-30 py-24 cursor-pointer transition-transform hover:scale-[1.02] active:scale-[0.99] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
                >
                  <svg
                    width="28"
                    height="28"
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="var(--text-light)"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M10 13V3M10 3L5.5 7.5M10 3l4.5 4.5" />
                    <path d="M3.5 14.5V16a1 1 0 001 1h11a1 1 0 001-1v-1.5" />
                  </svg>
                  <span style={{ color: "var(--text-light)", fontWeight: 500 }}>
                    Drop an image, or click to browse
                  </span>
                  {/* <span
                    className="text-xs"
                    style={{
                      color: "var(--text-light)",
                      fontFamily: "var(--font-commanding)",
                    }}
                  >
                    jpg · png · webp
                  </span> */}
                </button>
                <p
                  className="mx-auto text-md font-light leading-relaxed drop-shadow-md"
                  style={{
                    width: "380px",
                    color: "var(--text-light)",
                    textAlign: "justify",
                    textAlignLast: "center",
                  }}
                >
                  GeoCoach interprets natural surroundings such as vegetation,
                  terrain, and climate to figure out where you are. Upload an
                  image and the app explains what those clues tell us about the
                  location.
                </p>
                <SupportedMap />
              </div>
            ) : loading ? (
              <div className="flex-1" />
            ) : (
              <div className="flex flex-col gap-8 max-w-xl mx-auto w-full">
                {/* === HERO: the combined verdict === */}
                {countries.length > 0 && (
                  <div className="text-left">
                    <p
                      className="text-xs uppercase tracking-[0.15em] mb-2"
                      style={{
                        color: "var(--text-muted)",
                        fontFamily: "var(--font-commanding)",
                      }}
                    >
                      Most likely location
                    </p>
                    <p
                      className="text-4xl mb-3"
                      style={{
                        fontFamily: "var(--font-headline)",
                        color: "var(--text-light)",
                      }}
                    >
                      {countries[0]}
                    </p>

                    {reasoning && (
                      <p
                        className="text-sm leading-relaxed"
                        style={{ color: "var(--text-muted)" }}
                      >
                        {reasoning}
                      </p>
                    )}

                    {countries.length > 1 && (
                      <div className="mt-4">
                        <p
                          className="text-xs uppercase tracking-[0.15em] mb-2"
                          style={{
                            color: "var(--text-muted)",
                            fontFamily: "var(--font-commanding)",
                          }}
                        >
                          Other likely candidates
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {countries.slice(1).map((c) => (
                            <span
                              key={c}
                              className="text-xs px-2.5 py-1 rounded-full"
                              style={{
                                color: "var(--text-light)",
                                background: "rgba(245,240,230,0.14)",
                                border: "1px solid rgba(245,240,230,0.28)",
                              }}
                            >
                              {c}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* === SECONDARY: the raw image-model guess, for comparison === */}
                {predictedGuesses.length > 0 && (
                  <div className="text-left">
                    <p
                      className="text-xs uppercase tracking-[0.15em] mb-2"
                      style={{
                        color: "var(--text-muted)",
                        fontFamily: "var(--font-commanding)",
                      }}
                    >
                      Our model's guess
                    </p>
                    <div className="flex flex-col gap-2">
                      {predictedGuesses.slice(0, 5).map((g) => {
                        const pct = Math.round(g.confidence * 100);
                        return (
                          <div
                            key={g.country}
                            className="flex items-center gap-3"
                          >
                            <span
                              className="text-xs w-28 shrink-0 truncate"
                              style={{ color: "var(--text-muted)" }}
                            >
                              {g.country}
                            </span>
                            <div
                              className="flex-1 h-1 rounded-full overflow-hidden"
                              style={{ background: "rgba(255,255,255,0.15)" }}
                            >
                              <div
                                className="h-full rounded-full"
                                style={{
                                  width: `${pct}%`,
                                  background: "var(--accent)",
                                }}
                              />
                            </div>
                            <span
                              className="text-xs w-8 text-right shrink-0"
                              style={{
                                fontFamily: "var(--font-commanding)",
                                color: "var(--text-muted)",
                              }}
                            >
                              {pct}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* === EVIDENCE: individual clues that fed into the verdict === */}
                {(() => {
                  const clues = [
                    ...Object.entries(vision).map(
                      ([category, predictions]) => ({
                        label: predictions[0]?.label,
                        pct: Math.round(
                          (predictions[0]?.confidence || 0) * 100,
                        ),
                        sub: category.replaceAll("_", " "),
                      }),
                    ),
                    ...(roadRaw?.any_markings_detected
                      ? (["white_line", "yellow_line"] as const)
                          .filter((k) => roadRaw[k])
                          .map((k) => ({
                            label: `${roadRaw[k]!.double_line ? "Double" : "Single"} ${roadRaw[k]!.pattern} ${k === "yellow_line" ? "yellow" : "white"} line`,
                            pct: Math.round(roadRaw[k]!.confidence * 100),
                            sub: "Road marking",
                          }))
                      : []),
                  ].filter((c) => c.label);

                  const sorted = [...clues].sort(
                    (a, b) => (b.pct || 0) - (a.pct || 0),
                  );
                  const top = sorted.filter((c) => (c.pct ?? 0) >= 50);
                  const rest = sorted.filter((c) => (c.pct ?? 0) < 50);

                  return (
                    clues.length > 0 && (
                      <div
                        className="p-8 text-left flex flex-col gap-4 rounded-2xl"
                        style={{
                          background: `
      radial-gradient(ellipse 70% 50% at 75% 20%, rgba(245,240,230,0.06) 0%, transparent 50%),
      radial-gradient(ellipse 55% 70% at 15% 80%, rgba(245,240,230,0.04) 0%, transparent 55%),
      linear-gradient(135deg, var(--bg-mid) 0%, var(--bg-deep) 100%)
    `,
                          border: "1px solid rgba(245,240,230,0.12)",
                          boxShadow: "0 4px 24px rgba(0,0,0,0.3)",
                        }}
                      >
                        <p
                          className="text-sm uppercase tracking-[0.15em] text-center"
                          style={{
                            color: "rgba(255, 255, 255, 0.85)",
                            fontFamily: "var(--font-commanding)",
                          }}
                        >
                          Features
                        </p>
                        {top.map((c, i) => (
                          <div key={i}>
                            <p
                              className="text-xs uppercase tracking-[0.15em] mb-0.5"
                              style={{
                                color: "rgba(255, 255, 255, 0.85)",
                                fontFamily: "var(--font-commanding)",
                              }}
                            >
                              {c.sub}
                            </p>
                            <div className="flex items-center justify-between gap-3">
                              <p
                                className="text-sm"
                                style={{ color: "rgba(255, 255, 255, 0.6)" }}
                              >
                                {c.label}
                              </p>
                              {c.pct !== null && (
                                <span
                                  className="text-xs shrink-0"
                                  style={{
                                    fontFamily: "var(--font-commanding)",
                                    color: "rgba(245,240,230,0.6)",
                                  }}
                                >
                                  {c.pct}%
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                        {rest.length > 0 && (
                          <details>
                            <summary
                              className="text-xs uppercase tracking-[0.06em] cursor-pointer select-none"
                              style={{
                                color: "rgba(255, 255, 255, 0.85)",
                                fontFamily: "var(--font-commanding)",
                              }}
                            >
                              Uncertain clues
                            </summary>
                            <div className="flex flex-col gap-3 mt-3">
                              {rest.map((c, i) => (
                                <div
                                  key={i}
                                  className="flex items-center justify-between gap-3"
                                >
                                  <span
                                    className="text-sm"
                                    style={{
                                      color: "rgba(255, 255, 255, 0.6)",
                                    }}
                                  >
                                    {c.label}
                                  </span>
                                  {c.pct !== null && (
                                    <span
                                      className="text-xs shrink-0"
                                      style={{
                                        fontFamily: "var(--font-commanding)",
                                        color: "rgba(245,240,230,0.5)",
                                      }}
                                    >
                                      {c.pct}%
                                    </span>
                                  )}
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )
                  );
                })()}
              </div>
            )}
          </div>
        </div>
      </div>
      {dragOverlay}
      {rateLimitOverlay}
    </main>
  );
}
