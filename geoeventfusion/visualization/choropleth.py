"""Source country choropleth map for GEOEventFusion.

Renders a Folium choropleth map with bubble markers sized by article volume.
Rendering only — no data transformation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# FIPS-to-ISO mapping for common countries (needed for centroid lookups)
# GDELT uses FIPS country codes; the geo_utils centroid table uses ISO 2-letter codes
_FIPS_TO_ISO2 = {
    "AF": "AF",  # Afghanistan
    "AO": "AO",  # Angola
    "AS": "AU",  # Australia (FIPS: AS)
    "AZ": "AZ",  # Azerbaijan
    "BA": "BA",  # Bosnia
    "BD": "BD",  # Bangladesh
    "BY": "BY",  # Belarus
    "CD": "CD",  # DR Congo
    "CF": "CF",  # Central African Republic
    "CH": "CN",  # China (FIPS: CH)
    "CG": "CG",  # Republic of Congo
    "CI": "CI",  # Ivory Coast
    "CM": "CM",  # Cameroon
    "CO": "CO",  # Colombia
    "DZ": "DZ",  # Algeria
    "EG": "EG",  # Egypt
    "ER": "ER",  # Eritrea
    "ET": "ET",  # Ethiopia
    "FR": "FR",  # France
    "GB": "GB",  # United Kingdom (FIPS: UK)
    "UK": "GB",  # United Kingdom
    "GE": "GE",  # Georgia
    "GH": "GH",  # Ghana
    "GN": "GN",  # Guinea
    "HT": "HT",  # Haiti
    "IN": "IN",  # India
    "IQ": "IQ",  # Iraq
    "IR": "IR",  # Iran
    "IS": "IL",  # Israel (FIPS: IS)
    "JA": "JP",  # Japan (FIPS: JA)
    "JP": "JP",  # Japan
    "KE": "KE",  # Kenya
    "KN": "KP",  # North Korea (FIPS: KN)
    "KS": "KR",  # South Korea (FIPS: KS)
    "KZ": "KZ",  # Kazakhstan
    "LB": "LB",  # Lebanon
    "LY": "LY",  # Libya
    "MA": "MA",  # Morocco
    "MD": "MD",  # Moldova
    "ML": "ML",  # Mali
    "MO": "MA",  # Morocco alternate
    "MR": "MR",  # Mauritania
    "MW": "MW",  # Malawi
    "MX": "MX",  # Mexico
    "MZ": "MZ",  # Mozambique
    "NG": "NG",  # Nigeria
    "NI": "NI",  # Nicaragua
    "NP": "NP",  # Nepal
    "PH": "PH",  # Philippines
    "PK": "PK",  # Pakistan
    "RS": "RS",  # Russia (FIPS: RS)
    "RU": "RU",  # Russia
    "RW": "RW",  # Rwanda
    "SA": "SA",  # Saudi Arabia
    "SF": "ZA",  # South Africa (FIPS: SF)
    "SO": "SO",  # Somalia
    "SS": "SS",  # South Sudan
    "SU": "SD",  # Sudan (FIPS: SU)
    "SY": "SY",  # Syria
    "TD": "TD",  # Chad
    "TJ": "TJ",  # Tajikistan
    "TN": "TN",  # Tunisia
    "TU": "TR",  # Turkey (FIPS: TU)
    "TZ": "TZ",  # Tanzania
    "UA": "UA",  # Ukraine
    "UG": "UG",  # Uganda
    "US": "US",  # United States
    "UZ": "UZ",  # Uzbekistan
    "VE": "VE",  # Venezuela
    "VM": "VN",  # Vietnam (FIPS: VM)
    "YE": "YE",  # Yemen
    "ZA": "ZM",  # Zambia (FIPS: ZA)
    "ZI": "ZW",  # Zimbabwe (FIPS: ZI)
}


def render_choropleth_map(
    country_stats: Any,
    title: str = "Coverage by Source Country",
    output_path: Optional[str | Path] = None,
) -> Optional[str | Path]:
    """Render a Folium bubble map of article coverage by source country.

    Args:
        country_stats: CountryStats dataclass with top_countries list.
        title: Map title string.
        output_path: If provided, save the map HTML to this path.

    Returns:
        Output path if saved, None otherwise.
    """
    try:
        import folium

        from geoeventfusion.utils.geo_utils import country_centroid
        from geoeventfusion.visualization.theme import (
            THEME_ACCENT,
            THEME_SPIKE,
        )

        if not country_stats or not country_stats.top_countries:
            logger.warning("Choropleth map: no country stats — skipping")
            return None

        # Create base map centered at 20°N, 0°E
        fmap = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles="CartoDB dark_matter",
        )

        # Add title as a map overlay
        title_html = (
            f'<div style="position:fixed;top:10px;left:50px;z-index:9999;'
            f'background-color:rgba(0,0,0,0.7);padding:8px 14px;border-radius:4px;'
            f'color:{THEME_ACCENT};font-size:14px;font-weight:bold;">{title}</div>'
        )
        fmap.get_root().html.add_child(folium.Element(title_html))

        # Compute max volume for scaling
        max_vol = max(
            (c.get("volume", 0) for c in country_stats.top_countries),
            default=1,
        )
        if max_vol == 0:
            max_vol = 1

        for country_entry in country_stats.top_countries:
            country_code = country_entry.get("country", "").upper()
            volume = country_entry.get("volume", 0)
            share = country_entry.get("share", 0)

            # Convert FIPS to ISO2 for centroid lookup
            iso2 = _FIPS_TO_ISO2.get(country_code, country_code)
            centroid = country_centroid(iso2)
            if not centroid:
                continue

            lat, lon = centroid
            radius = 5 + (volume / max_vol) * 20  # Scale radius 5–25px

            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=THEME_SPIKE,
                fill=True,
                fill_color=THEME_SPIKE,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"<b>{country_code}</b><br>Volume: {volume:.4f}<br>Share: {share:.1%}",
                    max_width=150,
                ),
                tooltip=f"{country_code}: {share:.1%}",
            ).add_to(fmap)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmap.save(str(output_path))
            logger.info("Saved choropleth map: %s", output_path)
            return output_path

        return None

    except ImportError:
        logger.warning("folium is required for choropleth map rendering")
        return None
    except Exception as exc:
        logger.error("Choropleth map rendering failed: %s", exc)
        return None
