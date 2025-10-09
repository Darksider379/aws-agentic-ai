import os
from PIL import Image


def logos(st) :
    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
    GCP_LOGO_PATH = os.path.join(ASSETS_DIR, "gcp.png")
    AZURE_LOGO_PATH = os.path.join(ASSETS_DIR, "azure.png")
    LOGO_BOX = 160  # px

    # def transparent(path: str, box: int):
    #     #gcp
    #     if path.__contains__("gcp.png"):
    #         gcp_image = Image.open(path).convert("RGBA")
    #         datas = gcp_image.getdata()
    #         bg = (255, 255, 255)      # background color to remove
    #         tol = 12                  # tolerance for near-white

    #         new_data = []
    #         for r, g, b, a in datas:
    #             if abs(r-bg[0]) <= tol and abs(g-bg[1]) <= tol and abs(b-bg[2]) <= tol:
    #                 new_data.append((r, g, b, 0))   # make fully transparent
    #             else:
    #                 new_data.append((r, g, b, a))

    #         gcp_image.putdata(new_data)
    #         gcp_image.thumbnail((box, box), Image.LANCZOS)
    #         gcp_image.save("gcp_image_trans.png")  # PNG keeps transparency
    #         return gcp_image

    #     #azure
    #     if path.__contains__("azure.png"):
    #         azure_image = Image.open(path).convert("RGBA")
    #         datas = azure_image.getdata()
    #         bg = (255, 255, 255)      # background color to remove
    #         tol = 12                  # tolerance for near-white

    #         new_data = []
    #         for r, g, b, a in datas:
    #             if abs(r-bg[0]) <= tol and abs(g-bg[1]) <= tol and abs(b-bg[2]) <= tol:
    #                 new_data.append((r, g, b, 0))   # make fully transparent
    #             else:
    #                 new_data.append((r, g, b, a))

    #         azure_image.putdata(new_data)
    #         azure_image.thumbnail((box, box), Image.LANCZOS)
    #         azure_image.save("azure_image_trans.png")  # PNG keeps transparency
    #         return azure_image

    # gcp_image = transparent(GCP_LOGO_PATH, LOGO_BOX)
    # azure_image = transparent(AZURE_LOGO_PATH, LOGO_BOX)

    def load_logo_square(path: str, box: int) -> Image.Image:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((box, box), Image.LANCZOS)
        canvas = Image.new("RGBA", (box, box), (0, 0, 0, 0))
        x = (box - img.width) // 2
        y = (box - img.height) // 2
        canvas.paste(img, (x, y), img)
        return canvas

    gcp_logo  = load_logo_square(GCP_LOGO_PATH, LOGO_BOX)
    az_logo   = load_logo_square(AZURE_LOGO_PATH, LOGO_BOX)

    col_gcp, col_az = st.columns(2)
    with col_gcp:
        st.image(gcp_logo, use_container_width=False)
        st.markdown("**Coming soon**")
        st.caption("Recommendations, anomalies, and forecasting for GKE, Cloud Storage, and BigQuery.")
    with col_az:
        st.image(az_logo, use_container_width=False)
        st.markdown("**Coming soon**")
        st.caption("AKS, Blob Storage, and Azure SQL optimization are planned for upcoming releases.")

    st.caption("© Agentic AI Hackathon • Bedrock Agents + Lambda + Athena")