"use client";

import WorldMap from "react-svg-worldmap";
import type { ISOCode } from "react-svg-worldmap";

const TRAINED_COUNTRY_CODES: string[] = [
  "SE",
  "BO",
  "BR",
  "GB",
  "US",
  "RU",
  "HU",
  "FR",
  "CA",
  "TH",
  "ES",
  "UZ",
  "MA",
  "BD",
  "NP",
  "MX",
  "KW",
  "LV",
  "DE",
  "IT",
  "PK",
  "ZM",
  "AR",
  "GR",
  "RW",
  "NZ",
  "DO",
  "JP",
  "FI",
  "SK",
  "PL",
  "PH",
  "VN",
  "SG",
  "NI",
  "TW",
  "BY",
  "IR",
  "TZ",
  "IS",
  "AL",
  "AU",
  "CO",
  "MY",
  "HK",
  "PY",
  "NO",
  "HN",
  "QA",
  "RO",
  "NL",
  "BG",
  "DK",
  "ZA",
  "ID",
  "LT",
  "CH",
  "LK",
  "IN",
  "IE",
  "PT",
  "NG",
  "CL",
  "BN",
  "MN",
  "OM",
  "BH",
  "DZ",
  "RS",
  "EE",
  "HR",
  "TR",
  "BA",
  "UG",
  "MD",
  "MM",
  "CY",
  "CZ",
  "AT",
  "SI",
  "BE",
  "EG",
  "IL",
  "GH",
  "PS",
  "GT",
  "TN",
  "TL",
  "RE",
  "EC",
  "MU",
  "SN",
  "UY",
  "JO",
  "AE",
  "ET",
  "GE",
  "KR",
  "PE",
  "SA",
  "MZ",
  "CN",
  "MK",
  "PA",
  "LS",
  "TM",
  "LU",
  "CR",
  "SL",
  "AZ",
  "ML",
  "LA",
  "KG",
  "XK",
  "KZ",
  "CD",
  "SV",
  "KE",
  "MR",
];

const countries = TRAINED_COUNTRY_CODES.map((code) => ({
  country: code.toLowerCase(),
  value: 1,
})) as { country: ISOCode; value: number }[];

export default function SupportedMap() {
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
            Geocoach is trained on road imagery from countries around the world.
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
              size="responsive"
              data={countries}
              tooltipTextFunction={(context: { countryName: string }) =>
                context.countryName
              }
              tooltipBgColor="rgba(18,25,42,0.95)"
              tooltipTextColor="#F5E6C8"
            />
          </div>
        </div>
      </div>
    </section>
  );
}
