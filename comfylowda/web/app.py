# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
#
# The Comfy Catapult project requires contributions made to this file be licensed
# under the MIT license or a compatible open source license. See LICENSE.md for
# the license text.

import hypercorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from hypercorn.asyncio import serve as hypercorn_serve

from .routes import router as api_router
from .settings import AppSettings


async def InitializeFastAPI(settings: AppSettings) -> FastAPI:
  app = FastAPI(title=settings.project_name,
                version=settings.version,
                description=settings.description,
                docs_url=settings.docs_url,
                redoc_url=settings.redoc_url,
                debug=settings.debug)
  app.state.settings = settings
  app.state.templates = Jinja2Templates(directory='comfylowda/web/templates')

  app.mount('/static',
            StaticFiles(directory='comfylowda/web/static'),
            name='static')

  # Add the routes to the app
  app.include_router(api_router)

  # Add CORS middleware
  app.add_middleware(
      CORSMiddleware,
      allow_origins=settings.allowed_hosts,
      allow_credentials=True,
      allow_methods=['*'],
      allow_headers=['*'],
  )
  # Get hostname -I from system
  # hostname = (await anyio.run_process("hostname -I")).stdout.strip().decode()
  # console.print(f"`hostname -I`: {hostname}", style="bold green")
  return app


async def RunFastAPI(*, app: FastAPI, settings: AppSettings):
  hconfig = hypercorn.config.Config()

  # hostnames = [(await anyio.run_process("hostname -I")).stdout.strip().decode()]

  bind = []
  # for hostname in hostnames:
  bind += [f'{settings.hostname}:{settings.port}']

  hconfig.bind = bind
  hconfig.loglevel = 'debug' if settings.debug else 'warning'

  await hypercorn_serve(app, hconfig)  # type: ignore
