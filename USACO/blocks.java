import java.util.*;
import java.io.*;
import java.lang.*;
public class blocks
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("blocks.in"));
      int num1 = in.nextInt(), num2 = in.nextInt(), num3 = in.nextInt();
      Box one = new Box(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE);
      Box two = new Box(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE);
      Box three = new Box(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE);
   
      ArrayList<Pair> b1 = new ArrayList<>();
      for(int i = 0; i < num1; i++)
      {
         int x = in.nextInt() + 20, y = in.nextInt() + 20;
         b1.add(new Pair(x, y));
         one.minx = Math.min(one.minx, x);
         one.maxx = Math.max(one.maxx, x);
         one.miny = Math.min(one.miny, y);
         one.maxy = Math.max(one.maxy, y);
      }
      
      ArrayList<Pair> b2 = new ArrayList<>();
      for(int i = 0; i < num2; i++)
      {
         int x = in.nextInt() + 20, y = in.nextInt() + 20;
         b2.add(new Pair(x, y));
         two.minx = Math.min(two.minx, x);
         two.maxx = Math.max(two.maxx, x);
         two.miny = Math.min(two.miny, y);
         two.maxy = Math.max(two.maxy, y);
      }
      
      ArrayList<Pair> b3 = new ArrayList<>();
      for(int i = 0; i < num3; i++)
      {
         int x = in.nextInt() + 20, y = in.nextInt() + 20;
         b3.add(new Pair(x, y));
         three.minx = Math.min(three.minx, x);
         three.maxx = Math.max(three.maxx, x);
         three.miny = Math.min(three.miny, y);
         three.maxy = Math.max(three.maxy, y);
      }
      
      Queue<State> q = new LinkedList<>();
      q.add(new State(0, 0, 0, 0));
      
      boolean[][][][] visited = new boolean[41][41][41][41];
      int[][][][] moves = new int[41][41][41][41];
      visited[20][20][20][20] = true;
      moves[20][20][20][20] = 0;
      
      int[] x1change = {1, -1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0};
      int[] y1change = {0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1};
      int[] x2change = {0, 0, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0};
      int[] y2change = {0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1};
      
      while(!q.isEmpty())
      {
         State current = q.remove();
         
         for(int i = 0; i < 12; i++)
         {
            State newstate = new State(current.dx1, current.dy1, current.dx2, current.dy2);
            newstate.dx1 += x1change[i];
            newstate.dy1 += y1change[i];
            newstate.dx2 += x2change[i];
            newstate.dy2 += y2change[i];
            
            if(bounds(newstate.dx1) && bounds(newstate.dy1) && bounds(newstate.dx2) && bounds(newstate.dy2))
            {
               if(isok(newstate, b1, b2, b3))
               {
                  if(nooverlap(newstate, b1, b2, b3, one, two, three))
                  {
                     moves[newstate.dx1 + 20][newstate.dy1 + 20][newstate.dx2 + 20][newstate.dy2 + 20] = moves[current.dx1 + 20][current.dy1 + 20][current.dx2 + 20][current.dy2 + 20] + 1;
                     System.out.println(moves[newstate.dx1 + 20][newstate.dy1 + 20][newstate.dx2 + 20][newstate.dy2 + 20]);
                     System.exit(0);
                  }
                  else if(!visited[newstate.dx1 + 20][newstate.dy1 + 20][newstate.dx2 + 20][newstate.dy2 + 20])
                  {
                     visited[newstate.dx1 + 20][newstate.dy1 + 20][newstate.dx2 + 20][newstate.dy2 + 20] = true;
                     moves[newstate.dx1 + 20][newstate.dy1 + 20][newstate.dx2 + 20][newstate.dy2 + 20] = moves[current.dx1 + 20][current.dy1 + 20][current.dx2 + 20][current.dy2 + 20] + 1;
                     q.add(newstate);
                  }
               }
            }
         }
      }
      
      System.out.println("-1");
   }
   
   static boolean nooverlap(State state, ArrayList<Pair> b1, ArrayList<Pair> b2, ArrayList<Pair> b3, Box one, Box two, Box three)
   {
      int minx1 = one.minx, minx2 = two.minx, minx3 = three.minx, miny1 = one.miny, miny2 = two.miny, miny3 = three.miny, maxx1 = one.maxx, maxx2 = two.maxx, maxx3 = three.maxx, maxy1 = one.maxy, maxy2 = two.maxy, maxy3 = three.maxy;
      
      boolean onetwo = false, twothree = false, threeone = false;
      if(minx2 + state.dx2 > maxx1 + state.dx1 || maxx2 + state.dx2 < minx1 + state.dx1 || miny2 + state.dy2 > maxy1 + state.dy1 || maxy2 + state.dy2 < miny2 + state.dy2)
         onetwo = true;
        
      if(minx3 > maxx2 + state.dx2 || maxx3 < minx2 + state.dx2 || miny3 > maxy2 + state.dy2 || maxy3 < miny2 + state.dy2)
         twothree = true;
        
      if(minx3 > maxx1 + state.dx1 || maxx3 < minx1 + state.dx1 || miny3 > maxy1 + state.dy1 || maxy3 < miny1 + state.dy1)
         threeone = true;
        
      return onetwo && twothree && threeone;
   }
   
   
   static void changebox(State newstate, Box a, Box b, Box c)
   {
      a.minx += newstate.dx1;
      a.maxx += newstate.dx1;
      a.miny += newstate.dy1;
      a.maxy += newstate.dy1;
        
      b.minx += newstate.dx2;
      b.maxx += newstate.dx2;
      b.miny += newstate.dy2;
      b.maxy += newstate.dy2;
   }
   
   static boolean isok(State newstate, ArrayList<Pair> b1, ArrayList<Pair> b2, ArrayList<Pair> b3)
   {
      boolean[][] temp = new boolean[41][41];
      for(Pair p : b3)
         temp[p.x][p.y] = true;
      
      for(Pair p : b1)
      {
         Pair newp = new Pair(p.x + newstate.dx1, p.y + newstate.dy1);
         if(!bounds(newp.x - 20) || !bounds(newp.y - 20)) 
            return false;
         if(temp[newp.x][newp.y]) 
            return false;
         temp[newp.x][newp.y] = true;
      }
      
      for(Pair p : b2)
      {
         Pair newp = new Pair(p.x + newstate.dx2, p.y + newstate.dy2);
         if(!bounds(newp.x - 20) || !bounds(newp.y - 20)) 
            return false;
         if(temp[newp.x][newp.y]) 
            return false;
         temp[newp.x][newp.y] = true;
      }
      
      return true;
   }
   
   static boolean bounds(int t)
   {
      return t >= -20 && t <= 20;
   }
   
   static class Pair
   {
      public int x;
      public int y;
      public Pair(int a, int b)
      {
         x = a;
         y = b;
      }
      
      public boolean equals(Pair p)
      {
         return x == p.x && y == p.y;
      }
      
      public String toString()
      {
         return x + " " + y;
      }
   }
   
   static class State
   {
      public int dx1;
      public int dy1;
      public int dx2;
      public int dy2;
      public State(int a, int b, int c, int d)
      {
         dx1 = a;
         dy1 = b;
         dx2 = c;
         dy2 = d;
      }
   }
   
   static class Box
   {
      public int minx, miny, maxx, maxy;
      public Box(int a, int b, int c, int d)
      {
         minx = a;
         miny = b;
         maxx = c;
         maxy = d;
      }
   }
}