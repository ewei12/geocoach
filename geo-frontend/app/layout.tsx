import type { Metadata } from "next";
import {
  Geist,
  Geist_Mono,
  Darumadrop_One,
  Outfit,
  Space_Mono,
} from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const darumadrop = Darumadrop_One({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-darumadrop",
});

const outfit = Outfit({
  weight: ["400", "500", "600", "700"],
  subsets: ["latin"],
  variable: "--font-body",
});

const spaceMono = Space_Mono({
  weight: ["400", "700"],
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "vienna",
  description: "lil pup",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Bpmf+Zihi+Kai+Std&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className={`${darumadrop.variable} ${outfit.variable} ${spaceMono.variable}`}
      >
        {children}
      </body>
    </html>
  );
}
