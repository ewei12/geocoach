// RateLimitOverlay.tsx
export function RateLimitOverlay({ message }: { message: string }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        backdropFilter: "blur(4px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          background: "white",
          borderRadius: 12,
          padding: "32px 40px",
          maxWidth: 420,
          textAlign: "center",
          boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
        }}
      >
        <h2 style={{ marginBottom: 12 }}>Demo limit reached.</h2>
        <p style={{ color: "#555", lineHeight: 1.5 }}>{message}</p>
      </div>
    </div>
  );
}