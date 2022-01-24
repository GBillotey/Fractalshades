# -*- coding: utf-8 -*-
"""
===========================================
Ultra-deep embedded Julia set 2
===========================================

This example shows the kind of structure that occur very deep in the Mandelbrot
set. The width of this image is only 2.e-2608 [#f2]_.
This is not only below the separation power of double, but the delta are also
way below the minimal magnitude that can be stored in a double
(around  1e-323). A ad-hoc dataype is used internally during the Series
approximation step.

The period of the central minibrot is 1351428 ; the use of Series
approximations allows to skip around 5 millions iterations.

As the running time for this script is more than an hour, this image
has been pre-computed.


.. [#f2] **Credit:** Coordinates courtesy of Microfractal :
        <https://www.deviantart.com/microfractal/art/Mandelbrot-Deep-Julia-Morphing-22-Golden-Sphere-886123409>
"""

import os
import numpy as np

import fractalshades as fs
import fractalshades.settings as fssettings

import fractalshades.models as fsm
import fractalshades.colors as fscolors
from fractalshades.postproc import (
    Postproc_batch,
    Continuous_iter_pp,
    Raw_pp,
    DEM_normal_pp,
    DEM_pp
)
from fractalshades.colors.layers import (
    Color_layer,
    Bool_layer,
    Blinn_lighting,
    Normal_map_layer,
)


def plot(directory):
    """
    Example plot of a very deep location
    Credit for the coordinates :
    # https://www.deviantart.com/microfractal/art/Mandelbrot-Deep-Julia-Morphing-27-889296737
    """
    
    precision = 5902
    nx = 400
    x = '-1.9404343391649050769962346475186072278319641808230662599205517910251916276548103321780393703002901511964229685394182180175774134579074626768577135274496793100414713715570335087325046556894396124893910233087783535707602025517847716040269743594368331689714572576879233883275935457605063603006894251881221522558201597106903385665482041597153496239317381073750156972806404213045058334881253959240904562990379125209036304014395017901678049827832681503688449468823983686453040159583982025070358652632802967402190655895597252452906247107447553580839774509772718414325694795181915000998741703331798501814489594826131779661619203451886007864154540412555573017310918467219433607110514160937423998777336905856620831505781879255692497907899553742597080807726260469612543175296706105332088333004251783914564607408198970629339745573249718174621236212359899441124181211690116917519467774406855794182297359466142519530819187253529329520986961436064833330823838131207364015242129885412006996769664129832352326710838555344171203288267654560559605474638670638621559291618385558579381881512964375217092335412416495737164049962609451008509713895024279637794336011343338272341868004573058238659570663874955885766325158225066211958233843437491337729975554251031805243267598121899897886806893712191989739304038864487284116843359189233153041818334113044992638989969177604145749068983723364877337948464312839037418495900282222791814474284012942409331921985941286429774387954332832191142756604058660268881072901022349037319946148656307717101419465702063444970137294151804725639169273341080325716095770752501861343175095318495461763371959681185567466329862129213019725734699950057286002215689780554086783638935869306307101045336613495205800277535792398823399775277409303906245445406562145045176457304201196964982768698205723811312689350580712729358921506330043155937299577959587411854279677628152366141736287338748252315489447507434922908254441088761883701293743047988766569562929904402181765249072882456915721188941767489645485347154619629107780534894921446528318763623109377325602973851839252448039755255833821881600939318009336129866260632923080860886045584741703605036527712786921555776214129736935837712781104203112502819262221588816211784254058271538059170428338071355064081074911401759537745649952071465940790574248736029248134527250743440835672108445866312412343646310776786299415302572812561682090840295793922157182437769773379540889562552137917872213506799075215106676463394273034318675564340386079788477205875163076667885936104156131893643107568860880998286703495483410847768177981083797939777469608555449028052151916092452698716015821800884480341826059715415744139201393944658588862639622252229198062368152818812679708639672962906491662387271731687156236610777365875726224643306736306177703235025051844716119639411254688662046612549571660934559589602120535680118594808108362876917575684527607195726035857658695671465349593820731329363382408025736443450566778016833983415696720479170853528524947100651629685050278140094987767455229238105406096188453481403870513771498872875774576069871453441292925549736573349453893848291714433455848208439732623769434162325701766067942022954614687346951348901410213312812321005607700490187826819637589035841506228095653186209699533008590171280925236803793992931687374559757817009990679349583765071469905650831010341262324528494971125504067311471129323867705150887935735695264325508760125310202114058309744063251415754690681022688593236238833395522988351431932293630705225918317094138708607459909821876557872637982016932579678127665326245439259402090017415132129039970821728245471520346344589010690870095221802135807626749711031564866297292890319741269432375955208006338808812938818681223591629287315981797177524383491150960531339563244181720124391109074178092639501063798096454947697873737671452641547636284467076651469567726190430418442425231955382715846453473181434331695290188683887178316999971401917345437311218444833523059309883025424108370964739481852569381480503929613260278072782704193519017063799117160782245659921158794178808421427048237962220254475928176715975510249209554863516157569259789926575121092922843270515699371165916327362319057491365672506177980230696169584731075923425594050809983340806493191240936349853728113739105368748425481628192143606363383299600875394324934950687149227695197152343900217929923383381883895900730452708914284620041806530541953503670133178318558738801882672466934047026158505626580518323689749018281031087800528319718634512594581712051218746139870769822983517799487848192457976689904229288538134953696177332442827145528551062351490607017773075533744399284683275379331930117220190490252830009620782899281919812206800630903358605532887627482651813683441222860082777939933718915011230974764991422119088952999588530698220382146948765976457585257686023241711944857660844681150397148397378302094778999646154625016566311684039629672873289797718611864992327749119044785614885484861945426232548196668690768705503040659985179582936472804186934909500706692838670954060652197872035554004089547210120032156964583570144394349602426205332535208238759803124623710817681976908406252187454391661262004162305505463133792652878550523645560735416746830655740991954120045132244550620735576161643289563124225352154241452191320138903931793216231454401747114385467758718427797564914539107701177775202450046825185222359356319308888906238204367863085327579585812261146545736730032660355267185560839493920274771568534550515205790699388671332912079711530787559659275713429257386953698754826819571910124237039173584061046841680435720047870205286150230214002696651458591217511425381426410948181231719759349374320099029069652085236587003255078375587127334895902227085138187526727379937379385440991480763330611682517781043243240892010247711956776923616156041194208818237338870677030235393138064693399711866285170497557298633766890835955351759984855000026999929000057000038'
    y = '0.0000001047568887763372509633832576737386299232828768467315643452729319743916529367776347789526219727053853956422837777429644367149992901297188698442472215000988363128248056483707836399499038285209527766999255385350530528717822930896120067286197603661607639822457512159023072666906339368187688277638114846297965472816624084498006650846140354848113932437573861267841906704415092255548891789629210361656953587826963190229154363364061086983859281766354928198319281998687626624243711891974479049001435561943809164785239929491680552418974225361752612970785059205758306417821606681324029658691323115257069441118010057890614519758821334312963956936562537513848520428003517992753896187434416047414432662908844693827167675562692892920702450372484427222566437823977930073610083976894335900029209326682367591301539341692451635671073436783986948287618638924353669535699705632651075525036364226957529561367811471770469551274897130934125107430200220668517930996370187898188920083408423691112474022453650140202936447612519896354341706858093146226888671152853715292986720920442984465075439713086475675395994520046545592789135669618025174396990839025561469685965961206130039278227758483637152529794252993694425215169284185280067939873146775240661049092931178234776493937384521781020126188130877874494382258482191312170011841552895832938913572596277930454286653271078348532468681509160320545804333835185168226721582612608616466163391340564490827715599492891065058123530515016574160005398301066765249431622848433735636361086974482839974519632710659899746244218260530186024675140612296444295042603281239520683342269760222330202907513786361953308348966770153503012731162476737408880759392168267487639673567402004824091571973914999526638648163764577289548310675403531668634293099655110717753899109254859117623523169936062936538466633073744811132148607057411222375289965858888577648876624007187387469295077518602074081254820023155818631314606919663286627035167888411408042964383523839253923099492967275046102735246228553621703681465121062253761463229634498437870088821568844775794438046474482364625995358482332331363902279438186664116210654664507218679691869719755823229898983764568667793151506685493363620482381607293387826242146937556206626351656989242852645265561874006245503838247629849473632140738947944315628602296731443041140653117234841933559695737528618100485532191936533093074895976080899981768165633097519383721514736338130597154082325010332502535225995705117170983570007660619501555682998032688183122006553767716639845860266794906065657872505611161702971261180111939493976201783620930286478754450793542708135028026306042410314798898334753029861951178120733338690992653127467138089562068148252175823648909431417803609393174009600397999822443673347302192472431446560114464724868894116381578959129102224390562311857731067233735930816186353411917100845873120710438011380233391118643234478089182803927696879219564396539773547698778668338511448704768422609831079474931376163600775938625351616116983044817048249100930605435085929430745804536555149420485030060693255930965852759286219240211497558907320658701493568139347530158394069804499404202879791397998515555877815119023561462008239580418631088019900737516112239062465438183199801064096004485464867666226317630080876565409590326701490279641712309013344363832211984389475760415956181319648024173419320554124761934480519970825440259025700981345426691897334184860926451249231082708318612480718567923978130047994691093089597234057439606067249966338228719913163551001745094238370729530132841475090251680610455293420088518901842285337455861797626259750116772296218884866980193926167452203128570077388310167208234356810209733594422099749205877535743979542688897127891004674759841224259463008800741423017332134346622378572471028527608560106816738822861521634252768986101379781261403353773747224229838955726450006617169463057721555434186842872752210725796685443517749462233702364579763659318676859578483123370889106204729870890127854572708276506147900213240735641971454943690532571954991987952826035121348284793623235953681657926635462472927988377644904710395574017338687445764168406825326711178412213695205219855944025135430017277950329645145465692267426179164096316426549527014003736195767190876034367649710931282090615367480063924365345300343201392973733854026834397628273913614449174616915057192712396526516380742956248580052475400516354149644481168234662180213617833225372265572665356039216765530387622004935083275080464712048639660372355502617201834245588694364156325455226280272031765932703392330573852147611258268976771672646089049508987812607108453311990276713944360391086362095243736185695559978216621357248691647534468947609195035118909861611866480752083744095522143586752701041681892471355155265000614463749358981756439532337793130984880201393575814829860493822127481595172583093664169758942067378247413346533183240605602797340914158775692310801042138777132320550005322741339477544573617374092763215604070649739315856705206143086256948109158597305249109306490400190273473883591854938663587414339459014603604097860132349322738497275997413707612772454496196998338019449446352470120956684517020077915007695353076602437363706014803438476304893354474389113281787982015589126546527132729894510279717564364246541486015208899442824236866888984813488811080938750218286072673850567037045343614746806573001983741398864036117207441474311137717272762856730457669327476621608455184765222160647034049931664610867559314144859732523516617500160849522964689670493267249760223633567334027827280565800203592165004359766771148197256937594851929333709691334297783369992167032299489911808456323548876450551308312543795621057899633901180287870243734208340362258941766719255456106356032857614388926094337044830967435088780878290614598100731492146224629746447573040591533808805028012267496264230677474453784157956710041087220647723137082231467102085477888605554327000000000000000000000000'
    dx = '6.e-4392'  # ~ 7.5 / zoom

    # Ball method 1 found period: xxxxx
    # 64 terms: SA stop 5430472
    # 32 terms SA stop 5387712 [(8.27376011, -1235)]
    # 8 terms : SA stop 5131152 -> 5460601... * 

    test_corner = False
    if test_corner:
        import mpmath
        mpmath.mp.dps = precision
        x_corner = mpmath.mpf(x) + 0.5 * mpmath.mpf(dx)
        y_corner = mpmath.mpf(y) + 0.5 * mpmath.mpf(dx)
        x = str(x_corner)
        y = str(y_corner)


    # Set to True if you only want to rerun the post-processing part
    fssettings.chunk_size = 50
    fssettings.skip_calc = False
    # Set to True to enable multi-processing
    fssettings.enable_multiprocessing = True
    fssettings.no_newton = True
    fs.settings.inspect_calc = True
    fs.settings.optimize_RAM = False
    
    calc_dzndc = True

    calc_name="deep"
    colormap = fscolors.cmap_register["autumn"]

    f = fsm.Perturbation_mandelbrot(directory)
    f.zoom(
            precision=precision,
            x=x,
            y=y,
            dx=dx,
            nx=nx,
            xy_ratio=16./9.,
            theta_deg=-27., 
            projection="cartesian",
            antialiasing=True)

    f.calc_std_div(
            datatype=np.complex128,
            calc_name=calc_name,
            subset=None,
            max_iter=20100100,
            M_divergence=1.e3,
            epsilon_stationnary=1.e-3,
            SA_params={"cutdeg": 8,
                       "err": 1.e-6
                       },
            interior_detect=False,
            calc_dzndc=calc_dzndc)

    f.run()

    # Plot the image
    pp = Postproc_batch(f, calc_name)
    pp.add_postproc("cont_iter", Continuous_iter_pp())
    pp.add_postproc("DEM", DEM_pp())
    pp.add_postproc("interior", Raw_pp("stop_reason", func="x != 1."))
    if calc_dzndc:
        pp.add_postproc("DEM_map", DEM_normal_pp(kind="potential"))

    plotter = fs.Fractal_plotter(pp)   
    plotter.add_layer(Bool_layer("interior", output=False))
    if calc_dzndc:
      plotter.add_layer(Normal_map_layer("DEM_map", max_slope=35, output=True))
    plotter.add_layer(Color_layer(
            "cont_iter",
            func="np.log(x)",
            colormap=colormap,
            probes_z=[0., .5], # [0., 4.5]
            probes_kind="relative",
            output=True
    ))

    plotter["cont_iter"].set_mask(
            plotter["interior"],
            mask_color=(0., 0., 0.)
    )
#    if calc_dzndc:
#        plotter["DEM_map"].set_mask(
#                plotter["interior"],
#                mask_color=(0., 0., 0.)
#        )

    # This is where we define the lighting (here 3 ccolored light sources)
    # and apply the shading
    light = Blinn_lighting(0.3, np.array([1., 1., 1.]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=400.,
        shininess=400.,
        angles=(90., 20.),
        coords=None,
        color=np.array([0.5, 0.5, .3]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=400.,
        shininess=400.,
        angles=(70., 30.),
        coords=None,
        color=np.array([0.5, 0.5, .3]))
    light.add_light_source(
        k_diffuse=0.2,
        k_specular=400.,
        shininess=400.,
        angles=(40., 20.),
        coords=None,
        color=np.array([0.5, 0.5, .3]))
    light.add_light_source(
        k_diffuse=1.8,
        k_specular=0.,
        shininess=0.,
        angles=(70., 20.),
        coords=None,
        color=np.array([1.0, 1.0, 1.0]))

    plotter["cont_iter"].shade(plotter["DEM_map"], light)
    

    plotter.plot()

def _plot_from_data(plot_dir):
    # Private function only used when building fractalshades documentation
    # This example takes too long too run to autogenerate the image for the
    # gallery each - so just grabbing the file from the html doc static path
    import PIL

    data_path = fs.settings.output_context["doc_data_dir"]
    im = PIL.Image.open(os.path.join(data_path, "gaia.jpg"))
    rgb_im = im.convert('RGB')
    tag_dict = {"Software": "fractalshades " + fs.__version__,
                "example_plot": "gaia"}
    pnginfo = PIL.PngImagePlugin.PngInfo()
    for k, v in tag_dict.items():
        pnginfo.add_text(k, str(v))
    if fs.settings.output_context["doc"]:
        fs.settings.add_figure(fs._Pillow_figure(rgb_im, pnginfo))
    else:
        # Should not happen
        raise RuntimeError()


if __name__ == "__main__":
    # Some magic to get the directory for plotting: with a name that matches
    # the file or a temporary dir if we are building the documentation
    try:
        realpath = os.path.realpath(__file__)
        plot_dir = os.path.splitext(realpath)[0]
        plot(plot_dir)
    except NameError:
        import tempfile
        with tempfile.TemporaryDirectory() as plot_dir:
            _plot_from_data(plot_dir)

